#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analysis tool for idefix/pluto/fargo3d simulations (in polar coordinates).
"""
# adapted from pbenitez-llambay, gwafflard-fernandez, cmt robert & glesur

import argparse
from collections import ChainMap
from multiprocessing import Pool
from pathlib import Path
import functools
import pkg_resources
import glob
import re
from typing import List, Optional
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import toml
import inifix
from inifix.format import iniformat
import lic
from copy import copy

from nonos.config import DEFAULTS
from nonos.logging import print_warn, print_err
from nonos.__version__ import __version__
from nonos.parsing import parse_output_number_range, parse_vmin_vmax, parse_image_format, is_set

# TODO: recheck in 3D
# TODO: check in plot function if corotate=True works for all vtk and dpl
#        (initial planet location) -> computation to calculate the grid rotation speed
# TODO: compute gas surface density and not just gas volume density :
#        something like self.data*=np.sqrt(2*np.pi)*self.aspectratio*self.xmed
# TODO: compute vortensity
# TODO: compute vertical flows (cf vertical_flows.txt)
# TODO: re-check if each condition works fine
# TODO: recheck the writeField feature
# TODO: streamline analysis: weird azimuthal reconnection ?
# TODO: streamline analysis: test if the estimation of the radial spacing works
# TODO: write a better way to save pictures (function in PlotNonos maybe)
# TODO: do not forget to change all the functions that use dpl (planet location),
#        which is valid if the planet is in a fixed cicular orbit
# TODO: test corotate in the (R,z) plane
# TODO: create a test that compares when midplane=False
#        (average=True+corotate=True) & (average=True+corotate=False) should be identical
# TODO: check how the class arguments (arg=None) are defined between different classes
# TODO: test averaging procedure (to compare with theroetical surface density profiles)
# TODO: think how to check is_averageSafe when average=True

class DataStructure:
    """
    Class that helps create the datastructure
    in the readtVTKPolar function
    """
    pass

def readVTKPolar(filename, field='RHO', cell='edges', computedata=True):
    """
    Adapted from Geoffroy Lesur
    Function that reads a vtk file in polar coordinates
    """
    nfound = len(glob.glob(filename))
    if nfound!=1:
        raise FileNotFoundError("In readVTKPolar: %s not found."%filename)

    fid=open(filename,"rb")

    # define our datastructure
    V=DataStructure()

    # raw data which will be read from the file
    V.data={}

    # datatype we read
    dt=np.dtype(">f")   # Big endian single precision floats

    s=fid.readline()    # VTK DataFile Version x.x
    s=fid.readline()    # Comments

    s=fid.readline()    # BINARY
    s=fid.readline()    # DATASET RECTILINEAR_GRID

    slist=s.split()
    grid_type=str(slist[1],'utf-8')
    if grid_type != "STRUCTURED_GRID":
        fid.close()
        raise ValueError("In readVTKPolar: Wrong VTK file type.\nCurrent type is: '%s'.\nThis routine can only open Polar VTK files."%(grid_type))

    s=fid.readline()    # DIMENSIONS NX NY NZ
    slist=s.split()
    V.nx=int(slist[1])
    V.ny=int(slist[2])
    V.nz=int(slist[3])
    # print("nx=%d, ny=%d, nz=%d"%(V.nx,V.ny,V.nz))

    s=fid.readline()    # POINTS NXNYNZ float
    slist=s.split()
    npoints=int(slist[1])
    points=np.fromfile(fid,dt,3*npoints)
    s=fid.readline()    # EXTRA LINE FEED

    V.points=points
    if V.nx*V.ny*V.nz != npoints:
        raise ValueError("In readVTKPolar: Grid size (%d) incompatible with number of points (%d) in the data set"%(V.nx*V.ny*V.nz,npoints))

    # Reconstruct the polar coordinate system
    x1d=points[::3]
    y1d=points[1::3]
    z1d=points[2::3]

    xcart=np.transpose(x1d.reshape(V.nz,V.ny,V.nx))
    ycart=np.transpose(y1d.reshape(V.nz,V.ny,V.nx))
    zcart=np.transpose(z1d.reshape(V.nz,V.ny,V.nx))

    r=np.sqrt(xcart[:,0,0]**2+ycart[:,0,0]**2)
    theta=np.unwrap(np.arctan2(ycart[0,:,0],xcart[0,:,0]))
    z=zcart[0,0,:]

    s=fid.readline()    # CELL_DATA (NX-1)(NY-1)(NZ-1)
    slist=s.split()
    data_type=str(slist[0],'utf-8')
    if data_type != "CELL_DATA":
        fid.close()
        raise ValueError("In readVTKPolar: this routine expect 'CELL DATA' as produced by PLUTO, not '%s'."%data_type)
    s=fid.readline()    # Line feed

    if cell=='edges':
        if V.nx>1:
            V.nx=V.nx-1
            V.x=r
        else:
            V.x=r
        if V.ny>1:
            V.ny=V.ny-1
            V.y=theta
        else:
            V.y=theta
        if V.nz>1:
            V.nz=V.nz-1
            V.z=z
        else:
            V.z=z

    # Perform averaging on coordinate system to get cell centers
    # The file contains face coordinates, so we extrapolate to get the cell center coordinates.
    elif cell=='centers':
        if V.nx>1:
            V.nx=V.nx-1
            V.x=0.5*(r[1:]+r[:-1])
        else:
            V.x=r
        if V.ny>1:
            V.ny=V.ny-1
            V.y=(0.5*(theta[1:]+theta[:-1])+np.pi)%(2.0*np.pi)-np.pi
        else:
            V.y=theta
        if V.nz>1:
            V.nz=V.nz-1
            V.z=0.5*(z[1:]+z[:-1])
        else:
            V.z=z

    if computedata:
        while 1:
            s=fid.readline() # SCALARS/VECTORS name data_type (ex: SCALARS imagedata unsigned_char)
            #print repr(s)
            if len(s)<2:         # leave if end of file
                break
            slist=s.split()
            datatype=str(slist[0],'utf-8')
            varname=str(slist[1],'utf-8')
            if datatype == "SCALARS":
                fid.readline()  # LOOKUP TABLE
                V.data[varname] = np.transpose(np.fromfile(fid,dt,V.nx*V.ny*V.nz).reshape(V.nz,V.ny,V.nx))
            elif datatype == "VECTORS":
                Q=np.fromfile(fid,dt,3*V.nx*V.ny*V.nz)

                V.data[varname+'_X']=np.transpose(Q[::3].reshape(V.nz,V.ny,V.nx))
                V.data[varname+'_Y']=np.transpose(Q[1::3].reshape(V.nz,V.ny,V.nx))
                V.data[varname+'_Z']=np.transpose(Q[2::3].reshape(V.nz,V.ny,V.nx))

            else:
                raise ValueError("In readVTKPolar: Unknown datatype '%s', should be 'SCALARS' or 'VECTORS'" % datatype)
                break

            if varname!=field:
                if datatype == "SCALARS":
                    del V.data[varname]
                elif datatype == "VECTORS":
                    del V.data[varname+'_X'], V.data[varname+'_Y'], V.data[varname+'_Z']

            fid.readline()  #extra line feed
    fid.close()
       
    return V

class Parameters():
    """
    Adapted from Pablo Benitez-Llambay
    Class for reading the simulation parameters.
    input: string -> name of the parfile, normally *.ini
    """
    def __init__(self, config, directory=None, paramfile=None, corotate=None, isPlanet=None):
        self.config = copy(config)

        if directory is not None:
            self.config['datadir'] = directory
        if corotate is not None:
            self.config['corotate'] = corotate
        if isPlanet is not None:
            self.config['isPlanet'] = isPlanet

        self.paramfile = paramfile

    def load(self):
        lookup_table = {
            "idefix.ini" : "idefix",
            "pluto.ini": "pluto",
            "variables.par": "fargo3d",
        }
        if self.paramfile is None:
            found = {paramfile: Path(self.config['datadir']).joinpath(paramfile).is_file() for paramfile in lookup_table}
            nfound = sum(list(found.values()))
            if nfound == 0:
                raise FileNotFoundError("idefix.ini, pluto.ini or variables.par not found.")
            elif nfound > 1:
                raise RuntimeError("found more than one possible ini file.")
            self.paramfile = list(lookup_table.keys())[list(found.values()).index(True)]
        elif self.paramfile not in lookup_table:
            raise FileNotFoundError("For now, impossible to choose your parameter file.\nBy default, the code searches idefix.ini, pluto.ini or variables.par.")

        self.code = lookup_table[self.paramfile]
        self.iniconfig = inifix.load(os.path.join(self.config['datadir'], self.paramfile))

        if self.code=='idefix':
            self.data_files = list(glob.glob1(self.config['datadir'], "data.*.vtk"))
            # self.h0 = self.iniconfig["Setup"]["h0"]
            if self.config["isPlanet"]:
                if Path(self.config["datadir"]).joinpath("planet0.dat").is_file():
                    with open('planet0.dat','r') as f1:
                        datafile = f1.readlines()
                        self.qpl = np.array([float(line.split()[7]) for line in datafile])
                        self.dpl = np.array([np.sqrt(float(line.split()[1])**2+float(line.split()[2])**2+float(line.split()[3])**2) for line in datafile])
                        self.xpl = np.array([float(line.split()[1]) for line in datafile])
                        self.ypl = np.array([float(line.split()[2]) for line in datafile])
                        self.tpl = np.array([float(line.split()[8]) for line in datafile])
                else:
                    self.qpl = np.array([self.iniconfig["Planet"]["qpl"] for _ in self.data_files])
                    self.dpl = np.array([self.iniconfig["Planet"]["dpl"] for _ in self.data_files])
                self.omegaplanet = np.sqrt((1.0+self.qpl)/self.dpl/self.dpl/self.dpl)

            if self.config["corotate"]:
                self.vtk = self.iniconfig["Output"]["vtk"]
                if self.config["isPlanet"]:
                    self.omegagrid = self.omegaplanet
                else:
                    self.omegagrid = np.zeros(len(self.data_files))

        elif self.code=='pluto':
            self.data_files = list(glob.glob1(self.config["datadir"],"data.*.vtk"))
            # self.h0 = 0.05
            if self.config["isPlanet"]:
                self.qpl = np.full(len(self.data_files), self.iniconfig["Parameters"]["Mplanet"]/self.iniconfig["Parameters"]["Mstar"])
                print_warn("Initial distance not defined in pluto.ini.\nBy default, dpl=1.0 for the computation of omegaP\n")
                self.dpl = np.ones(len(self.data_files))
                self.omegaplanet = np.sqrt((1.0+self.qpl)/self.dpl/self.dpl/self.dpl)

            if self.config["corotate"]:
                self.vtk = self.iniconfig["Static Grid Output"]["vtk"][0]
                if self.config["isPlanet"]:
                    self.omegagrid = self.omegaplanet
                else:
                    self.omegagrid = np.zeros(len(self.data_files))

        elif self.code=='fargo3d':
            self.data_files = [fn for fn in glob.glob1(self.config["datadir"], "gasdens*.dat") if re.match(r"gasdens\d+.dat", fn)]
            nfound = len(glob.glob1(self.config["datadir"],"*.cfg"))
            if nfound==0:
                raise FileNotFoundError("*.cfg file (FARGO3D planet parameters) does not exist in '%s' directory"%self.config["datadir"])
            elif nfound>1:
                raise RuntimeError("found more than one possible .cfg file.")

            cfgfile = glob.glob1(self.config["datadir"],"*.cfg")[0]

            self.cfgconfig = inifix.load(os.path.join(self.config["datadir"],cfgfile))
            # self.h0 = self.iniconfig["ASPECTRATIO"]
            if self.config["isPlanet"]:
                if Path(self.config["datasdir"]).joinpath("planet0.dat").is_file():
                    with open('planet0.dat','r') as f1:
                        data = f1.readlines()
                    columns = np.array(data, dtype="float64").T
                    self.qpl = columns[7]
                    self.dpl = np.sqrt(np.sum(columns[1:4]**2, axis=0))
                    self.xpl = columns[1]
                    self.ypl = columns[2]
                    self.tpl = columns[8]
                else:
                    self.qpl = np.full(len(self.data_files), self.cfgconfig[list(self.cfgconfig)[0]][1])
                    self.dpl = np.full(len(self.data_files), self.cfgconfig[list(self.cfgconfig)[0]][0])
                self.omegaplanet = np.sqrt((1.0+self.qpl)/self.dpl/self.dpl/self.dpl)
            if self.config["corotate"]:
                self.vtk = self.iniconfig["NINTERM"]*self.iniconfig["DT"]
                if self.config["isPlanet"]:
                    self.omegagrid = self.omegaplanet
                else:
                    self.omegagrid = np.zeros(len(self.data_files))

        if not self.data_files:
            raise FileNotFoundError("No data files were found.")


class InitParamNonos(Parameters):
    def __init__(self, directory=None, paramfile=None, corotate=None, isPlanet=None):
        config_buffer = {} if paramfile is None else toml.load(paramfile)
        self.config = ChainMap(config_buffer, DEFAULTS)

        if directory is None:
            directory=self.config['datadir']
        self.directory=directory
        if corotate is None:
            corotate=self.config['corotate']
        self.corotate=corotate
        if isPlanet is None:
            isPlanet=self.config['isPlanet']
        self.isPlanet=isPlanet
        super().__init__(config=self.config, directory=self.directory, paramfile=paramfile, corotate=self.corotate, isPlanet=self.isPlanet) #All the Parameters attributes inside Field


class Mesh(Parameters):
    """
    Adapted from Pablo Benitez-Llambay
    Mesh class, for keeping all the mesh data.
    Input: directory [string] -> this is where the domain files are.
    """
    def __init__(self, config, directory=None, paramfile=None):
        super().__init__(config=config, directory=directory, paramfile=paramfile) #All the Parameters attributes inside Field
        super().load()
        if (self.code=='idefix' or self.code=='pluto'):
            domain=readVTKPolar(os.path.join(directory,'data.0000.vtk'), cell="edges", computedata=False)
            self.domain = domain

            self.nx = self.domain.nx
            self.ny = self.domain.ny
            self.nz = self.domain.nz

            self.xedge = self.domain.x #X-Edge
            self.yedge = self.domain.y-np.pi #Y-Edge
            self.zedge = self.domain.z #Z-Edge

            # index of the cell in the midplane
            self.imidplane = self.nz//2

        elif self.code=='fargo3d':
            nfound_x = len(glob.glob1(directory,"domain_x.dat"))
            if nfound_x!=1:
                raise FileNotFoundError("domain_x.dat not found.")
            nfound_y = len(glob.glob1(directory,"domain_y.dat"))
            if nfound_y!=1:
                raise FileNotFoundError("domain_y.dat not found.")
            nfound_z = len(glob.glob1(directory,"domain_z.dat"))
            if nfound_z!=1:
                raise FileNotFoundError("domain_z.dat not found.")

            domain_x = np.loadtxt(os.path.join(directory,"domain_x.dat"))
            #We avoid ghost cells
            domain_y = np.loadtxt(os.path.join(directory,"domain_y.dat"))[3:-3]
            domain_z = np.loadtxt(os.path.join(directory,"domain_z.dat"))
            if domain_z.shape[0]>6:
                domain_z=domain_z[3:-3]

            self.xedge = domain_y #X-Edge
            self.yedge = domain_x #Y-Edge
            # self.zedge = np.pi/2-domain_z #Z-Edge #latitute
            self.zedge = domain_z #Z-Edge #latitute

            self.nx=len(self.xedge)-1
            self.ny=len(self.yedge)-1
            self.nz=len(self.zedge)-1

            if np.sign(self.zedge[0])!=np.sign(self.zedge[-1]):
                self.imidplane = self.nz//2
            else:
                self.imidplane = -1

        self.xmed = 0.5*(self.xedge[1:]+self.xedge[:-1]) #X-Center
        self.ymed = 0.5*(self.yedge[1:]+self.yedge[:-1]) #Y-Center
        self.zmed = 0.5*(self.zedge[1:]+self.zedge[:-1]) #Z-Center

        # width of each cell in all directions
        self.dx = np.ediff1d(self.xedge)
        self.dy = np.ediff1d(self.yedge)
        self.dz = np.ediff1d(self.zedge)

        self.x = self.xedge
        self.y = self.yedge
        self.z = self.zedge

class FieldNonos(Mesh,Parameters):
    """
    Inspired by Pablo Benitez-Llambay
    Field class, it stores the mesh, parameters and scalar data
    for a scalar field.
    Input: field [string] -> filename of the field
           directory='' [string] -> where filename is
    """
    def __init__(self, init, directory=None, field=None, on=None, paramfile=None, diff=None, log=None, corotate=None, isPlanet=None, check=True):
        self.check=check
        self.init=init
        # TODO: simplify this block
        if directory is None:
            directory=self.init.directory
        if corotate is None:
            corotate=self.init.corotate
        if isPlanet is None:
            isPlanet=self.init.isPlanet
        Mesh.__init__(self, config=self.init.config, directory=directory, paramfile=paramfile)       #All the Mesh attributes inside Field
        Parameters.__init__(self, config=self.init.config, directory=directory, paramfile=paramfile, corotate=corotate, isPlanet=isPlanet) #All the Parameters attributes inside Field
        if on is None:
            on=self.init.config['on'][0]
        if field is None:
            field=self.init.config['field']
        if diff is None:
            diff=self.init.config['diff']
        if log is None:
            log=self.init.config['log']

        self.isPlanet = isPlanet
        self.corotate = corotate
        self.on = on
        self.diff = diff
        self.log = log

        self.field = field
        filedata = "data.%04d.vtk"%self.on
        filedata0 = "data.0000.vtk"
        if self.code=='pluto':
            self.field=self.field.lower()
        elif self.code=='fargo3d':
            if self.field=='RHO':
                self.field='dens'
            if self.field=='VX1':
                self.field='vy'
            if self.field=='VX2':
                self.field='vx'
            if self.field=='VX3':
                self.field='vz'
            filedata = "gas%s%d.dat"%(self.field,self.on)
            filedata0 = "gas%s0.dat"%self.field

        # FIXME: reactivate this warning
        #if self.corotate and not(self.isPlanet):
        #    warnings.warn("We don't rotate the grid if there is no planet for now.\nomegagrid = 0.")

        datafile = os.path.join(directory, filedata)

        if not os.path.isfile(datafile):
            raise FileNotFoundError(datafile)
        self.data = self.__open_field(datafile) #The scalar data is here.

        if self.diff:
            datafile = os.path.join(self.init.directory,filedata0)
            if not os.path.isfile(datafile):
                raise FileNotFoundError(datafile)
            self.data0 = self.__open_field(datafile)

        if self.log:
            if self.diff:
                self.data = np.log10(self.data/self.data0)
                self.title = r'log($\frac{%s}{%s_0}$)'%(self.field,self.field)
            else:
                self.data = np.log10(self.data)
                self.title = 'log(%s)'%self.field
        else:
            if self.diff:
                self.data = (self.data-self.data0)/self.data0
                self.title = r'$\frac{%s - %s_0}{%s_0}$'%(self.field,self.field,self.field)
            else:
                self.data = self.data
                self.title = '%s'%self.field

    def __open_field(self, f):
        """
        Reading the data
        """
        super().load()
        if(self.code=='idefix' or self.code=='pluto'):
            data = readVTKPolar(f, field=self.field, cell='edges').data[self.field].astype(np.float32)
            data = np.concatenate((data[:,self.ny//2:self.ny,:], data[:,0:self.ny//2,:]), axis=1)
        elif self.code=='fargo3d':
            data = np.fromfile(f, dtype='float32')
            data=(data.reshape(self.nz,self.nx,self.ny)).transpose(1,2,0) #rad, pĥi, theta
        """
        if we try to rotate a grid at 0 speed
        and if the domain is exactly [-pi,pi],
        impossible to perform the following calculation (try/except)
        we therefore don't move the grid if the rotation speed is null
        """
        if not(self.corotate and abs(self.vtk*sum(self.omegagrid[:self.on]))>1.0e-16):
            return data

        P,R = np.meshgrid(self.y,self.x)
        Prot=P-(self.vtk*sum(self.omegagrid[:self.on]))%(2*np.pi)
        try:
            index=(np.where(Prot[0]>np.pi))[0].min()
        except ValueError:
            index=(np.where(Prot[0]<-np.pi))[0].max()
        data=np.concatenate((data[:,index:self.ny,:],data[:,0:index,:]),axis=1)
        return data


class PlotNonos(FieldNonos):
    """
    Plot class which uses Field to compute different graphs.
    """
    def axiplot(self, ax, vmin=None, vmax=None, average=None, fontsize=None, **karg):
        if average is None:
            average=self.init.config['average']
        if average:
            dataRZ=np.mean(self.data,axis=1)
            dataR=np.mean(dataRZ,axis=1)*next(item for item in [self.z.max()-self.z.min(),1.0] if item!=0)
            dataProfile=dataR
        else:
            dataRZ=self.data[:,self.ny//2,:]
            dataR=dataRZ[:,self.imidplane]
            dataProfile=dataR
        vmin, vmax = parse_vmin_vmax(vmin, vmax, diff=self.diff, data=dataProfile)

        if fontsize is None:
            fontsize=self.init.config['fontsize']

        ax.plot(self.xmed,dataProfile,**karg)

        if not self.log:
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.set_ylim(vmin,vmax)
        ax.tick_params('both', labelsize=fontsize)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        ax.set_xlabel('Radius', fontsize=fontsize)
        ax.set_ylabel(self.title, fontsize=fontsize)
        # plt.legend(frameon=False)

    def plot(self, ax, vmin=None, vmax=None, midplane=None, geometry="cartesian", average=None, fontsize=None, cmap=None, **karg):
        """
        A layer for pcolormesh function.
        """
        vmin, vmax = parse_vmin_vmax(vmin, vmax, diff=self.diff, data=self.data)

        if midplane is None:
            midplane=self.init.config['midplane']
        if average is None:
            average=self.init.config['average']
        if fontsize is None:
            fontsize=self.init.config['fontsize']
        if cmap is None:
            cmap=self.init.config['cmap']

        zspan = self.z.ptp() or 1.0

        # (R,phi) plane
        if midplane:
            if self.x.shape[0]<=1:
                raise IndexError("No radial direction, the simulation is not 3D.\nTry midplane=False")
            if self.y.shape[0]<=1:
                raise IndexError("No azimuthal direction, the simulation is not 3D.\nTry midplane=False")
            if geometry == "cartesian":
                P,R = np.meshgrid(self.y,self.x)
                X = R*np.cos(P)
                Y = R*np.sin(P)
                if average:
                    im=ax.pcolormesh(X,Y,np.mean(self.data,axis=2)*zspan,
                              cmap=cmap,vmin=vmin,vmax=vmax,**karg)
                else:
                    im=ax.pcolormesh(X,Y,self.data[:,:,self.imidplane],
                              cmap=cmap,vmin=vmin,vmax=vmax,**karg)

                ax.set_aspect('equal')
                ax.xaxis.set_visible(True)
                ax.yaxis.set_visible(True)
                ax.set_ylabel('Y [c.u.]', family='monospace', fontsize=fontsize)
                ax.set_xlabel('X [c.u.]', family='monospace', fontsize=fontsize)
                if self.init.config['grid']:
                    ax.plot(X,Y,c='k',linewidth=0.07)
                    ax.plot(X.transpose(),Y.transpose(),c='k',linewidth=0.07)
            elif geometry == "polar":
                P,R = np.meshgrid(self.y,self.x)
                if average:
                    im=ax.pcolormesh(R,P,np.mean(self.data,axis=2)*zspan,
                              cmap=cmap,vmin=vmin,vmax=vmax,**karg)
                else:
                    im=ax.pcolormesh(R,P,self.data[:,:,self.imidplane],
                              cmap=cmap,vmin=vmin,vmax=vmax,**karg)

                ax.set_ylim(-np.pi,np.pi)
                ax.set_aspect('auto')
                ax.xaxis.set_visible(True)
                ax.yaxis.set_visible(True)
                ax.set_ylabel('Phi', family='monospace', fontsize=fontsize)
                ax.set_xlabel('Radius', family='monospace', fontsize=fontsize)
                if self.init.config['grid']:
                    ax.plot(R,P,c='k',linewidth=0.07)
                    ax.plot(R.transpose(),P.transpose(),c='k',linewidth=0.07)
            else:
                raise ValueError(f"Unknown geometry '{geometry}'")

            # ax.set_xlim(0.5,1.5)
            # ax.set_ylim(-0.8,0.8)
            ax.set_title(self.code, family='monospace', fontsize=fontsize)
            ax.tick_params('both', labelsize=fontsize)
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar=plt.colorbar(im, cax=cax, orientation='vertical')#, format='%.0e')
            cbar.ax.tick_params(labelsize=fontsize)
            cbar.set_label(self.title, family='monospace', fontsize=fontsize)

        # (R,z) plane
        else:
            if self.x.shape[0]<=1:
                raise IndexError("No radial direction, the simulation is not 3D.\nTry midplane=True")
            if self.z.shape[0]<=1:
                raise IndexError("No vertical direction, the simulation is not 3D.\nTry midplane=True")
            if geometry == "cartesian":
                Z,R = np.meshgrid(self.z,self.x)
                if average:
                    im=ax.pcolormesh(R,Z,np.mean(self.data,axis=1),
                              cmap=cmap,vmin=vmin,vmax=vmax,**karg)
                else:
                    im=ax.pcolormesh(R,Z,self.data[:,self.ny//2,:],
                              cmap=cmap,vmin=vmin,vmax=vmax,**karg)
                ax.set_aspect('auto')
                ax.xaxis.set_visible(True)
                ax.yaxis.set_visible(True)
                ax.set_ylabel('Z [c.u.]', family='monospace', fontsize=fontsize)
                ax.set_xlabel('X [c.u.]', family='monospace', fontsize=fontsize)
                # ax.set_xlim(-6.0,6.0)
                # ax.set_ylim(-6.0,6.0)
                if self.init.config['grid']:
                    # im=ax.scatter(X,Y,c=np.mean(self.data,axis=2))
                    ax.plot(R,Z,c='k',linewidth=0.07)
                    ax.plot(R.transpose(),Z.transpose(),c='k',linewidth=0.07)

                # ax.set_xlim(0.5,1.5)
                # ax.set_ylim(-0.8,0.8)
                ax.set_title(self.code, family='monospace', fontsize=fontsize)
                ax.tick_params('both', labelsize=fontsize)
                ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                ax.xaxis.set_ticks_position('both')
                ax.yaxis.set_ticks_position('both')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar=plt.colorbar(im, cax=cax, orientation='vertical')#, format='%.0e')
                cbar.ax.tick_params(labelsize=fontsize)
                cbar.set_label(self.title, family='monospace', fontsize=fontsize)
            else:
                Z,R = np.meshgrid(self.z,self.x)
                r = np.sqrt(R**2+Z**2)
                t = np.arctan2(R,Z)
                if average:
                    im=ax.pcolormesh(t,r,np.mean(self.data,axis=1),
                              cmap=cmap,vmin=vmin,vmax=vmax,**karg)
                else:
                    im=ax.pcolormesh(r,t,self.data[:,self.ny//2,:],
                              cmap=cmap,vmin=vmin,vmax=vmax,**karg)

                print_warn("Aspect ratio not defined for now.\nBy default, h0=0.05\n")
                tmin = np.pi/2-5*0.05
                tmax = np.pi/2+5*0.05
                # tmin = np.arctan2(1.0,Z.min())
                # tmax = np.arctan2(1.0,Z.max())

                """
                if polar plot in the (R,z) plane, use rather
                fig = plt.figure()
                ax = fig.add_subplot(111, polar=True)
                """
                ax.set_rmax(R.max())
                ax.set_theta_zero_location('N')
                ax.set_theta_direction(-1)
                ax.set_thetamin(tmin*180/np.pi)
                ax.set_thetamax(tmax*180/np.pi)

                ax.set_aspect('auto')
                ax.xaxis.set_visible(True)
                ax.yaxis.set_visible(True)
                ax.set_ylabel('Theta', family='monospace', fontsize=fontsize)
                ax.set_xlabel('Radius', family='monospace', fontsize=fontsize)
                # ax.set_xlim(-6.0,6.0)
                # ax.set_ylim(-6.0,6.0)
                if self.init.config['grid']:
                    # im=ax.scatter(X,Y,c=np.mean(self.data,axis=2))
                    ax.plot(r,t,c='k',linewidth=0.07)
                    ax.plot(r.transpose(),t.transpose(),c='k',linewidth=0.07)

                # ax.set_xlim(0.5,1.5)
                # ax.set_ylim(-0.8,0.8)
                ax.set_title(self.code, family='monospace', fontsize=fontsize)
                ax.tick_params('both', labelsize=fontsize)
                ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                ax.xaxis.set_ticks_position('both')
                ax.yaxis.set_ticks_position('both')
                cbar=plt.colorbar(im, orientation='vertical')#, format='%.0e')
                cbar.ax.tick_params(labelsize=fontsize)
                cbar.set_label(self.title, family='monospace', fontsize=fontsize)

class StreamNonos(FieldNonos):
    """
    Adapted from Pablo Benitez-Llambay
    Class which uses Field to compute streamlines.
    """
    def __init__(self, init, directory="", field=None, on=None, check=True):
        FieldNonos.__init__(self,init=init,field=field,on=on,directory=directory, check=check) #All the Parameters attributes inside Field

        if field is None:
            field=self.init.config['field']
        if on is None:
            on=self.init.config['on'][0]

    def bilinear(self,x,y,f,p):
        """
        Bilinear interpolation.
        Parameters
        ----------
        x = (x1,x2); y = (y1,y2)
        f = (f11,f12,f21,f22)
        p = (x,y)
        where x,y are the interpolated points and
        fij are the values of the function at the
        points (xi,yj).
        Output
        ------
        f(p): Float.
              The interpolated value of the function f(p) = f(x,y)
        """
        xp  = p[0]
        yp   = p[1]
        x1  = x[0]
        x2  = x[1]
        y1  = y[0]
        y2  = y[1]
        f11 = f[0]
        f12 = f[1]
        f21 = f[2]
        f22 = f[3]
        t = (xp-x1)/(x2-x1)
        u = (yp-y1)/(y2-y1)
        return (1.0-t)*(1.0-u)*f11 + t*(1.0-u)*f12 + t*u*f22 + u*(1-t)*f21

    def get_v(self, v, x, y):
        """
        For a real set of coordinates (x,y), returns the bilinear
        interpolated value of a Field class.
        """

        i = find_nearest(self.x,x)
        # i = int(np.log10(x/self.x.min())/np.log10(self.x.max()/self.x.min())*self.nx)
        # i = int((x-self.x.min())/(self.x.max()-self.x.min())*self.nx)
        j = int((y-self.y.min())/(self.y.max()-self.y.min())*self.ny)

        if i<0 or j<0 or i>v.shape[0]-2 or j>v.shape[1]-2:
            return None

        f11 = v[i,j,self.imidplane]
        f12 = v[i,j+1,self.imidplane]
        f21 = v[i+1,j,self.imidplane]
        f22 = v[i+1,j+1,self.imidplane]
        try:
            x1  = self.x[i]
            x2  = self.x[i+1]
            y1  = self.y[j]
            y2  = self.y[j+1]
            return self.bilinear((x1,x2),(y1,y2),(f11,f12,f21,f22),(x,y))
        except IndexError:
            return None

    def euler(self, vx, vy, x, y, reverse):
        """
        Euler integrator for computing the streamlines.
        Parameters:
        ----------

        x,y: Floats.
             Initial condition
        reverse: Boolean.
                 If reverse is true, the integration step is negative.

        Output
        ------

        (dx,dy): (float,float).
                 Are the azimutal and radial increments.
                 Only works for cylindrical coordinates.
        """
        sign = 1.0
        if reverse:
            sign = -1
        vr = self.get_v(vx,x,y)
        vt = self.get_v(vy,x,y)
        if None in (vt, vr):
            #Avoiding problems...
            return None,None

        l = np.min((((self.x.max()-self.x.min())/self.nx),((self.y.max()-self.y.min())/self.ny)))
        h = 0.5*l/np.sqrt((vr**2+vt**2))

        return sign*h*np.array([vr,vt/x])

    def get_stream(self, vx, vy, x0, y0, nmax=1000000, maxlength=4*np.pi, bidirectional=True, reverse=False):
        """
        Function for computing a streamline.
        Parameters:
        -----------

        x0,y0: Floats.
              Initial position for the stream
        nmax: Integer.
              Maxium number of iterations for the stream.
        maxlength: Float
                   Maxium allowed length for a stream
        bidirectional=True
                      If it's True, the stream will be forward and backward computed.
        reverse=False
                The sign of the stream. You can change it mannualy for a single stream,
                but in practice, it's recommeneded to use this function without set reverse
                and setting bidirectional = True.

        Output:
        -------

        If bidirectional is False, the function returns a single array, containing the streamline:
        The format is:

                                          np.array([[x],[y]])

        If bidirectional is True, the function returns a tuple of two arrays, each one with the same
        format as bidirectional=False.
        The format in this case is:

                                (np.array([[x],[y]]),np.array([[x],[y]]))

        This format is a little bit more complicated, and the best way to manipulate it is with iterators.
        For example, if you want to plot the streams computed with bidirectional=True, you can do:

        stream = get_stream(x0,y0)
        ax.plot(stream[0][0],stream[0][1]) #Forward
        ax.plot(stream[1][0],stream[1][1]) #Backward

        """

        if bidirectional:
            s0 = self.get_stream(vx, vy, x0, y0, reverse=False, bidirectional=False, nmax=nmax,maxlength=maxlength)
            s1 = self.get_stream(vx, vy, x0, y0, reverse=True,  bidirectional=False, nmax=nmax,maxlength=maxlength)
            return (s0,s1)

        l = 0
        x = [x0]
        y = [y0]

        for i in range(nmax):
            ds = self.euler(vx, vy, x0, y0, reverse=reverse)
            if ds[0] is None:
                # if(len(x)==1):
                #     print_warn("There was an error getting the stream, ds is NULL (see get_stream).")
                break
            l += np.sqrt(ds[0]**2+ds[1]**2)
            dx = ds[0]
            dy = ds[1]
            if np.sqrt(dx**2+dy**2)<1e-13:
                print_warn("(get_stream): ds is very small, check if you're in a stagnation point.\nTry selecting another initial point.")
                break
            if l > maxlength:
                # print("maxlength reached: ", l)
                break
            x0 += dx
            y0 += dy
            x.append(x0)
            y.append(y0)

        return np.array([x,y])

    def get_random_streams(self, vx, vy, xmin=None, xmax=None, ymin=None, ymax=None, n=30, nmax=100000):
        if xmin is None:
            xmin = self.x.min()
        if ymin is None:
            ymin = self.y.min()
        if xmax is None:
            xmax = self.x.max()
        if ymax is None:
            ymax = self.y.max()

        X = xmin + np.random.rand(n)*(xmax-xmin)
        # X = xmin*pow((xmax/xmin),np.random.rand(n))
        Y = ymin + np.random.rand(n)*(ymax-ymin)

        streams = []
        cter = 0
        for x,y in zip(X,Y):
            stream = self.get_stream(vx, vy, x, y, nmax=nmax, bidirectional=True)
            streams.append(stream)
            cter += 1
        return streams

    def get_fixed_streams(self, vx, vy, xmin=None, xmax=None, ymin=None, ymax=None, n=30, nmax=100000):
        if xmin is None:
            xmin = self.x.min()
        if ymin is None:
            ymin = self.y.min()
        if xmax is None:
            xmax = self.x.max()
        if ymax is None:
            ymax = self.y.max()

        X = xmin + np.linspace(0,1,n)*(xmax-xmin)
        # X = xmin*pow((xmax/xmin),np.random.rand(n))
        Y = ymin + np.linspace(0,1,n)*(ymax-ymin)

        streams = []
        cter2 = 0
        for x,y in zip(X,Y):
            stream = self.get_stream(vx, vy, x, y, nmax=nmax, bidirectional=True)
            streams.append(stream)
            cter2 += 1
        return streams

    def plot_streams(self, ax, streams, midplane=True, geometry="cartesian", **kargs):
        for stream in streams:
            for sub_stream in stream:
                # sub_stream[0]*=unit_code.length/unit.AU
                if midplane:
                    if geometry == "cartesian":
                        ax.plot(sub_stream[0]*np.cos(sub_stream[1]),sub_stream[0]*np.sin(sub_stream[1]),**kargs)
                    elif geometry == "polar":
                        ax.plot(sub_stream[0],sub_stream[1],**kargs)
                    else:
                        raise ValueError(f"Unknown geometry '{geometry}'")
                else:
                    if self.check:
                        raise NotImplementedError("For now, we do not compute streamlines in the (R,z) plane")

    def get_lic_streams(self, vx, vy):
        get_lic=lic.lic(vx[:,:,self.imidplane],vy[:,:,self.imidplane],length=30)
        return get_lic

    def plot_lic(self, ax, streams, midplane=True, geometry="cartesian", **kargs):
        if midplane:
            if geometry == "cartesian":
                P,R = np.meshgrid(self.y,self.x)
                X = R*np.cos(P)
                Y = R*np.sin(P)
                ax.pcolormesh(X,Y,streams,**kargs)
            elif geometry == "polar":
                P,R = np.meshgrid(self.y,self.x)
                ax.pcolormesh(R,P,streams,**kargs)
            else:
                raise ValueError(f"Unknown geometry '{geometry}'")
        else:
            if self.check:
                raise NotImplementedError("For now, we do not compute streamlines in the (R,z) plane")

def is_averageSafe(sigma0,sigmaSlope,plot=False):
    init = InitParamNonos() # initialize the major parameters
    fieldon = FieldNonos(init, field='RHO', on=0) # fieldon object with the density field at on=0
    datarz=np.mean(fieldon.data,axis=1) # azimuthally-averaged density field
    error=(sigma0*pow(fieldon.xmed,-sigmaSlope)-np.mean(datarz, axis=1)*next(item for item in [fieldon.z.max()-fieldon.z.min(),1.0] if item!=0))/(sigma0*pow(fieldon.xmed,-sigmaSlope)) # comparison between Sigma(R) profile and integral of rho(R,z) between zmin and zmax
    if any(100*abs(error)>3):
        print("With a maximum of %.1f percents of error, the averaging procedure may not be safe.\nzmax/h is probably too small.\nUse rather average=False (-noaverage) or increase zmin/zmax."%np.max(100*abs(error)))
    else:
        print("Only %.1f percents of error maximum in the averaging procedure."%np.max(100*abs(error)))
    if plot:
        fig, ax = plt.subplots()
        ax.plot(fieldon.xmed, np.mean(datarz, axis=1)*next(item for item in [fieldon.z.max()-fieldon.z.min(),1.0] if item!=0), label=r'$\int_{z_{min}}^{z_{max}} \rho(R,z)dz$ = (z$_{max}$-z$_{min}$)$\langle\rho\rangle_z$')
        ax.plot(fieldon.xmed, sigma0*pow(fieldon.xmed,-sigmaSlope), label=r'$\Sigma_0$R$^{-\sigma}$')
        # ax.plot(fieldon.xmed, np.mean(datarz, axis=1)*(fieldon.z.max()-fieldon.z.min()), label='integral of data using mean and zmin/zmax')
        # ax.plot(fieldon.xmed, sigma0*pow(fieldon.xmed,-sigmaSlope), label='theoretical reference')
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(True)
        ax.set_ylabel(r'$\Sigma_0(R)$', family='monospace', fontsize=10)
        ax.set_xlabel('Radius', family='monospace', fontsize=10)
        ax.tick_params('both', labelsize=10)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.legend(frameon=False, prop={'size': 10, 'family': 'monospace'})
        fig2, ax2 = plt.subplots()
        ax2.plot(fieldon.xmed, abs(error)*100)
        ax2.xaxis.set_visible(True)
        ax2.yaxis.set_visible(True)
        ax2.set_ylabel(r'Error (%)', family='monospace', fontsize=10)
        ax2.set_xlabel('Radius', family='monospace', fontsize=10)
        ax2.tick_params('both', labelsize=10)
        ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax2.xaxis.set_ticks_position('both')
        ax2.yaxis.set_ticks_position('both')
        plt.show()

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    barobj = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, barobj, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

# process function for parallisation purpose with progress bar
# counterParallel = Value('i', 0) # initialization of a counter
def process_field(on, init, dim, field, mid, geometry, avr, diff, log, corotate, stype, srmin, srmax, nstream, vmin, vmax, ft, cmap, isPlanet, pbar, parallel, directory, show:bool, dpi:int, fmt:str):
    ploton=PlotNonos(init, field=field, on=on, diff=diff, log=log, corotate=corotate, isPlanet=isPlanet, directory=directory, check=False)

    fig = plt.figure(figsize=(9,8))
    if polar := (geometry != "cartesian" and not mid):
        print_warn('plot not optimized for now in the (R,z) plane in polar.\nCheck in cartesian coordinates to be sure')
    ax = fig.add_subplot(111, polar=polar)

    # plot the field
    if dim == 2:
        ploton.plot(ax, vmin=vmin, vmax=vmax, midplane=mid, geometry=geometry, average=avr, fontsize=ft, cmap=cmap)
        if is_set(stype):
            streamon=StreamNonos(init, field=field, on=on, directory=directory, check=False)
            vx1on, vx2on = (FieldNonos(init, field=f'VX{i}', on=on, diff=False, log=False, corotate=corotate, isPlanet=isPlanet, directory=directory, check=False) for i in (1, 2))

            vr = vx1on.data
            vphi = vx2on.data
            if isPlanet:
                vphi -= vx2on.omegaplanet[on]*vx2on.xmed[:,None,None]
            if stype=="lic":
                streams=streamon.get_lic_streams(vr,vphi)
                streamon.plot_lic(ax,streams,cartesian=geometry=="cartesian", cmap='gray', alpha=0.3)
            else:
                kwargs = dict(vx=vr,vy=vphi,xmin=srmin,xmax=srmax, n=nstream)
                if stype=="random":
                    streams=streamon.get_random_streams(**kwargs)
                elif stype=="fixed":
                    streams=streamon.get_fixed_streams(**kwargs)
                else:
                    raise ValueError(f"Received unknown stype '{stype}'.")
                streamon.plot_streams(ax,streams,cartesian=geometry=="cartesian",color='k', linewidth=2, alpha=0.5)
        prefix = "Rphi" if mid else "Rz"


    # plot the 1D profile
    elif dim == 1:
        ploton.axiplot(ax, vmin=vmin, vmax=vmax, average=avr, fontsize=ft)
        prefix = "axi"
    filename = f"{prefix}_{field}{'_diff' if diff else ''}{'_log' if log else ''}{geometry if dim==2 else ''}{on:04d}.{fmt}"

    if show:
        plt.show()
    else:
        fig.savefig(filename, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:

    parser = argparse.ArgumentParser(prog='nonos',
                                     description=__doc__,
                                     )


    parser.add_argument(
        '-dir', dest='datadir',
        help=f"location of output files and param files (default: '{DEFAULTS['datadir']}').",
        )
    parser.add_argument(
        '-field',
        choices=['RHO', 'VX1', 'VX2', 'VX3'],
        help=f"name of field to plot (default: '{DEFAULTS['field']}').",
        )
    parser.add_argument(
        '-vmin',
        type=float,
        help=f"min value in -diff mode (default: {DEFAULTS['vmin']})",
        )
    parser.add_argument(
        '-vmax',
        type=float,
        help=f"max value in -diff mode (default: {DEFAULTS['vmin']})",
        )
    parser.add_argument(
        '-cpu', '-ncpu', dest="ncpu",
        type=int,
        help=f"number of parallel processes (default: {DEFAULTS['ncpu']}).",
    )

    select_group = parser.add_mutually_exclusive_group()
    select_group.add_argument(
        '-on',
        type=int,
        nargs='+',
        help="output number(s) (on) to plot. "
        "This can be a single value or a range (start, end, [step]) where both ends are inclusive. "
        "(default: last output available).",
        )
    select_group.add_argument(
        '-all',
        action="store_true",
        help="save an image for every available snapshot (this will force show=False)."
    )

    # boolean flags use False as a default value (...by default)
    # forcing them to None instead will allow them to be pruned
    # when we build the ChainMap config
    flag_group = parser.add_argument_group("boolean flags")
    flag_group.add_argument(
        '-diff',
        action="store_true",
        default=None,
        help="plot the relative perturbation of the field f, i.e. (f-f0)/f0.",
        )
    flag_group.add_argument(
        '-log',
        action="store_true",
        default=None,
        help="plot the log10 of the field f, i.e. log(f).",
        )
    flag_group.add_argument(
        '-isp', dest="isPlanet",
        action="store_true",
        default=None,
        help="is there a planet in the grid ?",
        )
    flag_group.add_argument(
        '-corotate',
        action="store_true",
        default=None,
        help="does the grid corotate? Works in pair with -isp.",
        )
    flag_group.add_argument(
        '-grid',
        action="store_true",
        default=None,
        help="show the computational grid."
    )
    flag_group.add_argument(
        '-streamlines',
        action="store_true",
        default=None,
        help="plot streamlines.",
        )
    flag_group.add_argument(
        '-rz',
        action="store_true",
        default=None,
        help="2D plot in the (R-z) plane (default: represent the midplane).",
        )
    flag_group.add_argument(
        '-noavr', '-noaverage', dest="noaverage",
        action="store_true",
        default=None,
        help="do not perform averaging along the third dimension. "
        )
    flag_group.add_argument(
        '-pbar', dest="progressBar",
        action="store_true",
        default=None,
        help="display a progress bar",
        )

    stream_group = parser.add_argument_group("streamlines options")
    stream_group.add_argument(
        '-stype', "-streamtype", dest="streamtype",
        choices=["random","fixed","lic"],
        help=f"streamlines method (default: '{DEFAULTS['streamtype']}')",
        )
    stream_group.add_argument(
        '-srmin', dest="rminStream",
        type=float,
        help=f"minimum radius for streamlines computation (default: {DEFAULTS['rminStream']}).",
        )
    stream_group.add_argument(
        '-srmax', dest="rmaxStream",
        type=float,
        help=f"maximum radius for streamlines computation (default: {DEFAULTS['rmaxStream']}).",
        )
    stream_group.add_argument(
        '-sn', dest="nstreamlines",
        type=int,
        help=f"number of streamlines (default: {DEFAULTS['nstreamlines']}).",
        )

    geom_group = parser.add_mutually_exclusive_group()
    geom_group.add_argument(
        "-geom", dest="geometry",
        choices=["cartesian", "polar"],
        )
    geom_group.add_argument(
        "-pol",
        action="store_true",
        help="shortcut for -geom=polar",
        )

    parser.add_argument(
        "-dim", dest="dimensionality",
        type=int,
        choices=[1, 2],
        help="dimensionality in projection: 1 for a line plot, 2 (default) for a map.",
    )
    parser.add_argument(
        '-ft', dest='fontsize',
        type=float,
        help=f"fontsize in the graph (default: {DEFAULTS['fontsize']}).",
        )
    parser.add_argument(
        '-cmap',
        help=f"choice of colormap for the -dim 2 maps (default: '{DEFAULTS['cmap']}').",
        )
    parser.add_argument(
        '-fmt', '-format', dest="format",
        help=f"select output image file format (default: {DEFAULTS['format']})"
    )
    parser.add_argument(
        '-dpi',
        type=int,
        help=f"image file resolution (default: DEFAULTS['dpi'])",
        )

    cli_only_group = parser.add_argument_group("CLI-only options")
    cli_input_group = cli_only_group.add_mutually_exclusive_group()
    cli_input_group.add_argument(
        "-input", "-i", dest="input",
        type=str,
        help="specify a configuration file."
    )
    cli_input_group.add_argument(
        "-isolated",
        action="store_true",
        help="ignore any existing 'nonos.toml' file."
    )
    cli_action_group = cli_only_group.add_mutually_exclusive_group()
    cli_action_group.add_argument(
        "-d", "-display", dest="display",
        action="store_true",
        help="open a graphic window with the plot (only works with a single image)"
    )
    cli_action_group.add_argument(
        "-version", "--version",
        action="store_true",
        help="show raw version number and exit",
    )
    cli_action_group.add_argument(
        "-logo",
        action="store_true",
        help="show Nonos logo with version number, and exit."
    )
    cli_action_group.add_argument(
        '-config',
        action="store_true",
        help="show configuration and exit."
    )
    cli_debug_group = cli_only_group.add_mutually_exclusive_group()
    cli_debug_group.add_argument(
        '-memory',
        action="store_true",
        help="show current memory usage and memory peak."
    )

    clargs = vars(parser.parse_args(argv))

    if clargs.pop("memory"):
	    memory_switch = True
    else:        
        memory_switch = False

    if memory_switch:
	    import tracemalloc
	    tracemalloc.start()

    # special cases: destructively consume CLI-only arguments with dict.pop
    if clargs.pop("pol"):
        clargs["geometry"] = "polar"

    if clargs.pop('logo'):
        with open(pkg_resources.resource_filename("nonos", "logo.txt")) as fh:
            logo = fh.read()
        print(f"{logo}{__doc__}Version {__version__}")
        return 0

    if clargs.pop('version'):
        print(__version__)
        return 0

    if clargs.pop("isolated"):
        config_file_args = {}
    elif (ifile := clargs.pop("input")) is not None:
        if not os.path.isfile(ifile):
            print_err(f"Couldn't find requested input file '{ifile}'.")
            return 1
        config_file_args = toml.load(ifile)
    elif os.path.isfile("nonos.toml"):
        config_file_args = toml.load("nonos.toml")
    else:
        config_file_args = {}

    init = InitParamNonos(directory=clargs['datadir'])

    # check that every parameter in the configuration is also exposed to the CLI
    assert not set(DEFAULTS).difference(set(clargs))

    # squeeze out any unset value form cli config to leave room for file parameters
    clargs = {k: v for k,v in clargs.items() if v is not None}

    # NOTE: init.config is also a ChainMap instance with a default layer
    # this may be seen either as hyperstatism (good thing) or error prone redundancy (bad thing)
    args = ChainMap(clargs, config_file_args, DEFAULTS)

    if clargs.pop("config"):
        conf_repr = {}
        for key in DEFAULTS:
            conf_repr[key] = args[key]
        print(f"# Generated with nonos {__version__}")
        print(iniformat(toml.dumps(conf_repr)))
        return 0

    try:
        init.load()
    except (FileNotFoundError,RuntimeError,ValueError) as exc:
        print_err(exc)
        return 1

    available = set(int(re.search(r"\d+", fn).group()) for fn in init.data_files)
    if args.pop('all'):
        requested = available
    else:
        try:
            requested = set(parse_output_number_range(args['on'], maxval=max(available)))
        except ValueError as exc:
            print_err(exc)
            return 1

    if not (toplot := list(requested.intersection(available))):
        print_err(f"No requested output file was found (requested {requested}, found {available}).")
        return 1
    args['on'] = toplot

    if (show := clargs.pop('display')) and len(args['on']) > 1:
        print_warn("display mode can not be used with multiple images, turning it off.")
        show = False

    if not show:
        try:
            args["format"] = parse_image_format(args["format"])
        except ValueError as exc:
            print_err(exc)
            return 1

    # check that every CLI-only argument was consumed at this point
    assert not set(clargs).difference(set(DEFAULTS))

    args['field'] = args['field'].upper()

    if args['rz'] and is_set(args['streamtype']):
        print_err("For now, we do not compute streamlines in the (R,z) plane")
        return 1

    if args['corotate'] and not args['isPlanet']:
        print_warn("We don't rotate the grid if there is no planet for now.\nomegagrid = 0.")

    if args['streamtype']=='lic':
        print_warn("TODO: check what is the length argument in StreamNonos().get_lic_streams ?")


    ref_on = args['on'][len(args['on'])//2]
    fieldon = FieldNonos(init, field=args['field'], on=ref_on, directory=args['datadir'], diff=args['diff'], check=False)
    if args['dimensionality'] == 2:
        data = fieldon.data
    elif args['dimensionality'] == 1:
        data = np.mean(np.mean(fieldon.data,axis=1),axis=1)

    vmin, vmax = parse_vmin_vmax(args['vmin'], args['vmax'], args['diff'], data)

    if args['ncpu'] > (ncpu := min(args['ncpu'], os.cpu_count())):
        print_warn(f"Requested {args['ncpu']}, but the runner only has access to {ncpu}.")

    if args['progressBar']:
        from rich.progress import track
    else:
        # replace rich.progress.track with a no-op dummy
        def track(iterable, *args, **kwargs):
            return iterable

    # call of the process_field function, whether it be in parallel or not
    func = functools.partial(
            process_field,
            init=init,
            dim=args["dimensionality"],
            field=args["field"],
            mid=not args["rz"],
            geometry=args["geometry"],
            avr=not args["noaverage"],
            diff=args["diff"],
            log=args["log"],
            corotate=args["corotate"],
            stype=args["streamtype"],
            srmin=args["rminStream"],
            srmax=args["rmaxStream"],
            nstream=args["nstreamlines"],
            vmin=vmin,
            vmax=vmax,
            ft=args["fontsize"],
            cmap=args["cmap"],
            isPlanet=args["isPlanet"],
            pbar=args["progressBar"],
            parallel=args["ncpu"] > 1,
            directory=args["datadir"],
            show=show,
            dpi=args["dpi"],
            fmt=args["format"],
        )

    tstart = time.time()
    with Pool(ncpu) as pool:
        list(track(pool.imap(func, args['on']), description="Processing snapshots", total=len(args['on'])))
    if not show:
        print(f"Operation took {time.time() - tstart:.2f}s")
    if memory_switch:
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage is {(current / 10**6):.2f}MB; Peak was {(peak / 10**6):.2f}MB")
        tracemalloc.stop()
    return 0
