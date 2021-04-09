#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
adapted from pbenitez-llambay, gwafflard-fernandez, cmt robert & glesur
"""

from multiprocessing import Pool, Value
from pathlib import Path
from shutil import copyfile
import functools
import pkg_resources
import glob
from typing import List, Optional
import argparse
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import toml
import inifix as ix
from rich import print as rprint
from rich.progress import track
import lic

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

def readVTKPolar(filename, cell='edges'):
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

        fid.readline()  #extra line feed
    fid.close()

    return V

class Parameters:
    """
    Adapted from Pablo Benitez-Llambay
    Class for reading the simulation parameters.
    input: string -> name of the parfile, normally *.ini
    """
    def __init__(self, config, directory=None, paramfile=None, corotate=None, isPlanet=None):
        if directory is None:
            directory=config['dir']
        if corotate is None:
            corotate=config['corotate']
        if isPlanet is None:
            isPlanet=config['isPlanet']
        if paramfile is None:
            lookup_table = {
                "idefix.ini" : "idefix",
                "pluto.ini": "pluto",
                "variables.par": "fargo3d",
            }
            found = {paramfile: Path(directory).joinpath(paramfile).is_file() for paramfile in lookup_table}
            nfound = sum(list(found.values()))
            if nfound == 0:
                raise FileNotFoundError("idefix.ini, pluto.ini or variables.par not found.")
            elif nfound > 1:
                raise RuntimeError("found more than one possible ini file.")
            paramfile = list(lookup_table.keys())[list(found.values()).index(True)]
            self.code = lookup_table[paramfile]
        else:
            raise FileNotFoundError("For now, impossible to choose your parameter file.\nBy default, the code searches idefix.ini, pluto.ini or variables.par.")

        self.paramfile = paramfile
        self.iniconfig = ix.load(os.path.join(directory,self.paramfile))

        if self.code=='idefix':
            self.n_file = len(glob.glob1(directory,"data.*.vtk"))
            # self.h0 = self.iniconfig["Setup"]["h0"]
            if isPlanet:
                if Path(directory).joinpath("planet0.dat").is_file():
                    with open('planet0.dat','r') as f1:
                        datafile = f1.readlines()
                        self.qpl = np.array([float(line.split()[7]) for line in datafile])
                        self.dpl = np.array([np.sqrt(float(line.split()[1])**2+float(line.split()[2])**2+float(line.split()[3])**2) for line in datafile])
                        self.xpl = np.array([float(line.split()[1]) for line in datafile])
                        self.ypl = np.array([float(line.split()[2]) for line in datafile])
                        self.tpl = np.array([float(line.split()[8]) for line in datafile])
                else:
                    self.qpl = np.array([self.iniconfig["Planet"]["qpl"] for i in range(self.n_file)])
                    self.dpl = np.array([self.iniconfig["Planet"]["dpl"] for i in range(self.n_file)])
                self.omegaplanet = np.sqrt((1.0+self.qpl)/self.dpl/self.dpl/self.dpl)

            if corotate:
                self.vtk = self.iniconfig["Output"]["vtk"]
                if isPlanet:
                    self.omegagrid = self.omegaplanet
                else:
                    self.omegagrid = np.array([0.0 for i in range(self.n_file)])

        elif self.code=='pluto':
            self.n_file = len(glob.glob1(directory,"data.*.vtk"))
            # self.h0 = 0.05
            if isPlanet:
                self.qpl = np.array([self.iniconfig["Parameters"]["Mplanet"]/self.iniconfig["Parameters"]["Mstar"] for i in range(self.n_file)])
                print_warn("Initial distance not defined in pluto.ini.\nBy default, dpl=1.0 for the computation of omegaP\n")
                self.dpl = np.array([1.0 for i in range(self.n_file)])
                self.omegaplanet = np.sqrt((1.0+self.qpl)/self.dpl/self.dpl/self.dpl)

            if corotate:
                self.vtk = self.iniconfig["Static Grid Output"]["vtk"][0]
                if isPlanet:
                    self.omegagrid = self.omegaplanet
                else:
                    self.omegagrid = np.array([0.0 for i in range(self.n_file)])

        elif self.code=='fargo3d':
            self.n_file = len(glob.glob1(directory,"gasdens*.dat")) - len(glob.glob1(directory,"gasdens*_*.dat"))
            nfound = len(glob.glob1(directory,"*.cfg"))
            if nfound==0:
                raise FileNotFoundError("*.cfg file (FARGO3D planet parameters) does not exist in '%s' directory"%directory)
            elif nfound>1:
                raise RuntimeError("found more than one possible .cfg file.")

            cfgfile = glob.glob1(directory,"*.cfg")[0]

            self.cfgconfig = ix.load(os.path.join(directory,cfgfile))
            # self.h0 = self.iniconfig["ASPECTRATIO"]
            if isPlanet:
                if Path(directory).joinpath("planet0.dat").is_file():
                    with open('planet0.dat','r') as f1:
                        datafile = f1.readlines()
                        self.qpl = np.array([float(line.split()[7]) for line in datafile])
                        self.dpl = np.array([np.sqrt(float(line.split()[1])**2+float(line.split()[2])**2+float(line.split()[3])**2) for line in datafile])
                        self.xpl = np.array([float(line.split()[1]) for line in datafile])
                        self.ypl = np.array([float(line.split()[2]) for line in datafile])
                        self.tpl = np.array([float(line.split()[8]) for line in datafile])
                else:
                    self.qpl = np.array([self.cfgconfig[list(self.cfgconfig)[0]][1] for i in range(self.n_file)])
                    self.dpl = np.array([self.cfgconfig[list(self.cfgconfig)[0]][0] for i in range(self.n_file)])
                self.omegaplanet = np.sqrt((1.0+self.qpl)/self.dpl/self.dpl/self.dpl)
            if corotate:
                self.vtk = self.iniconfig["NINTERM"]*self.iniconfig["DT"]
                if isPlanet:
                    self.omegagrid = self.omegaplanet
                else:
                    self.omegagrid = np.array([0.0 for i in range(self.n_file)])

        if self.n_file==0:
            raise FileNotFoundError("No data files (e.g., 'data.*.vtk' or 'gasdens*.dat') are found.")

class AnalysisNonos:
    """
    read the .toml file
    find parameters in config.toml (same directory as script)
    compute the number of data.*.vtk files in working directory
    """
    def __init__(self, directory_of_script=None, info=False):
        if directory_of_script is None:
            config_file = pkg_resources.resource_filename("nonos", "config.toml")
        else:
            config_file = os.path.join(directory_of_script, 'config.toml')

        self.config = toml.load(config_file)

        if info:
            print('--------------------------------------')
            print(toml.dumps(self.config))
            print('--------------------------------------')

class InitParamNonos:
    """
    Call the AnalysisNonos class to define the config dictionary
    and use it to call the Parameters class to initialize important parameters.
    """
    def __init__(self, directory=None, directory_of_script=None, info=False, paramfile=None, corotate=None, isPlanet=None):
        AnalysisNonos.__init__(self, directory_of_script=directory_of_script, info=info)
        if directory is None:
            directory=self.config['dir']
        self.directory=directory
        if corotate is None:
            corotate=self.config['corotate']
        self.corotate=corotate
        if isPlanet is None:
            isPlanet=self.config['isPlanet']
        self.isPlanet=isPlanet
        Parameters.__init__(self, config=self.config, directory=self.directory, paramfile=paramfile, corotate=self.corotate, isPlanet=self.isPlanet) #All the Parameters attributes inside Field
        if info:
            print(self.code.upper(), "analysis")

        if (self.code=='idefix' or self.code=='pluto'):
            domain=readVTKPolar(os.path.join(self.directory,'data.0000.vtk'), cell="edges")
            list_keys=list(domain.data.keys())
            if info:
                print("\nWORKS IN POLAR COORDINATES")
                print("Possible fields: ", list_keys)
                print('nR=%d, np=%d, nz=%d' % (domain.nx,domain.ny,domain.nz))

        elif self.code=='fargo3d':
            nfound_x = len(glob.glob1(self.directory,"domain_x.dat"))
            if nfound_x!=1:
                raise FileNotFoundError("domain_x.dat not found.")
            nfound_y = len(glob.glob1(self.directory,"domain_y.dat"))
            if nfound_y!=1:
                raise FileNotFoundError("domain_y.dat not found.")
            nfound_z = len(glob.glob1(self.directory,"domain_z.dat"))
            if nfound_z!=1:
                raise FileNotFoundError("domain_z.dat not found.")

            domain_x = np.loadtxt(os.path.join(self.directory,"domain_x.dat"))
            #We avoid ghost cells
            domain_y = np.loadtxt(os.path.join(self.directory,"domain_y.dat"))[3:-3]
            domain_z = np.loadtxt(os.path.join(self.directory,"domain_z.dat"))
            if domain_z.shape[0]>6:
                domain_z=domain_z[3:-3]

            if info:
                print("\nWORKS IN POLAR COORDINATES")
                print('nR=%d, np=%d, nz=%d' % (len(domain_y)-1,len(domain_x)-1,len(domain_z)-1))

class Mesh:
    """
    Adapted from Pablo Benitez-Llambay
    Mesh class, for keeping all the mesh data.
    Input: directory [string] -> this is where the domain files are.
    """
    def __init__(self, config, directory=None, paramfile=None):
        Parameters.__init__(self, config=config, directory=directory, paramfile=paramfile) #All the Parameters attributes inside Field
        if (self.code=='idefix' or self.code=='pluto'):
            domain=readVTKPolar(os.path.join(directory,'data.0000.vtk'), cell="edges")
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

class FieldNonos:
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
        if directory is None:
            directory=self.init.directory
        if corotate is None:
            corotate=self.init.corotate
        if isPlanet is None:
            isPlanet=self.init.isPlanet
        Mesh.__init__(self, config=self.init.config, directory=directory, paramfile=paramfile)       #All the Mesh attributes inside Field
        Parameters.__init__(self, config=self.init.config, directory=directory, paramfile=paramfile, corotate=corotate, isPlanet=isPlanet) #All the Parameters attributes inside Field
        if on is None:
            on=self.init.config['onStart']
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

        if self.check:
            if(not(self.isPlanet) and self.corotate):
                print_warn("We don't rotate the grid if there is no planet for now.\nomegagrid = 0.")

        nfdat = len(glob.glob1(self.init.directory,filedata))
        if nfdat!=1:
            raise FileNotFoundError(os.path.join(self.init.directory,filedata)+" not found")
        self.data = self.__open_field(os.path.join(self.init.directory,filedata)) #The scalar data is here.

        if self.diff:
            nfdat0 = len(glob.glob1(self.init.directory,filedata0))
            if nfdat0!=1:
                raise FileNotFoundError(os.path.join(self.init.directory,filedata0)+" not found")
            self.data0 = self.__open_field(os.path.join(self.init.directory,filedata0))

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
        if(self.code=='idefix' or self.code=='pluto'):
            data = readVTKPolar(f, cell='edges').data[self.field]
            data = np.concatenate((data[:,self.ny//2:self.ny,:], data[:,0:self.ny//2,:]), axis=1)
        elif self.code=='fargo3d':
            data = np.fromfile(f, dtype='float64')
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
    def __init__(self, init, directory="", field=None, on=None, diff=None, log=None, corotate=None, isPlanet=None, check=True):
        FieldNonos.__init__(self,init=init,field=field,on=on,directory=directory, diff=diff, log=log, corotate=corotate, isPlanet=isPlanet, check=check) #All the Parameters attributes inside Field

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
        if vmin is None:
            vmin=self.init.config['vmin']
            if not self.diff:
                vmin=dataProfile.min()
        if vmax is None:
            vmax=self.init.config['vmax']
            if not self.diff:
                vmax=dataProfile.max()
        if fontsize is None:
            fontsize=self.init.config['fontsize']

        if self.init.config['writeAxi']:
            axifile=open("axi%s%04d.csv"%(self.field.lower(),self.on),'w')
            for i in range(len(self.xmed)):
                axifile.write('%f,%f\n' %(self.xmed[i],dataProfile[i]))
            axifile.close()

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

    def plot(self, ax, vmin=None, vmax=None, midplane=None, cartesian=None, average=None, fontsize=None, cmap=None, **karg):
        """
        A layer for pcolormesh function.
        """
        if vmin is None:
            vmin=self.init.config['vmin']
            if not self.diff:
                vmin=self.data.min()
        if vmax is None:
            vmax=self.init.config['vmax']
            if not self.diff:
                vmax=self.data.max()
        if midplane is None:
            midplane=self.init.config['midplane']
        if cartesian is None:
            cartesian=self.init.config['cartesian']
        if average is None:
            average=self.init.config['average']
        if fontsize is None:
            fontsize=self.init.config['fontsize']
        if cmap is None:
            cmap=self.init.config['cmap']

        # (R,phi) plane
        if midplane:
            if self.x.shape[0]<=1:
                raise IndexError("No radial direction, the simulation is not 3D.\nTry midplane=False")
            if self.y.shape[0]<=1:
                raise IndexError("No azimuthal direction, the simulation is not 3D.\nTry midplane=False")
            if cartesian:
                P,R = np.meshgrid(self.y,self.x)
                X = R*np.cos(P)
                Y = R*np.sin(P)
                if average:
                    # next() function chooses ZMAX-ZMIN if 3D simulation, otherwise chooses 1.0
                    im=ax.pcolormesh(X,Y,np.mean(self.data,axis=2)*next(item for item in [self.z.max()-self.z.min(),1.0] if item!=0),
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
            else:
                P,R = np.meshgrid(self.y,self.x)
                if average:
                    im=ax.pcolormesh(R,P,np.mean(self.data,axis=2)*next(item for item in [self.z.max()-self.z.min(),1.0] if item!=0),
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
            if cartesian:
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
            on=self.init.config['onStart']

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
        if vt == None or vr == None: #Avoiding problems...
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

    def plot_streams(self, ax, streams, midplane=True, cartesian=True, **kargs):
        for stream in streams:
            for sub_stream in stream:
                # sub_stream[0]*=unit_code.length/unit.AU
                if midplane:
                    if cartesian:
                        ax.plot(sub_stream[0]*np.cos(sub_stream[1]),sub_stream[0]*np.sin(sub_stream[1]),**kargs)
                    else:
                        ax.plot(sub_stream[0],sub_stream[1],**kargs)
                else:
                    if self.check:
                        raise NotImplementedError("For now, we do not compute streamlines in the (R,z) plane")

    def get_lic_streams(self, vx, vy):
        get_lic=lic.lic(vx[:,:,self.imidplane],vy[:,:,self.imidplane],length=30)
        return get_lic

    def plot_lic(self, ax, streams, midplane=True, cartesian=True, **kargs):
        if midplane:
            if cartesian:
                P,R = np.meshgrid(self.y,self.x)
                X = R*np.cos(P)
                Y = R*np.sin(P)
                ax.pcolormesh(X,Y,streams,**kargs)
            else:
                P,R = np.meshgrid(self.y,self.x)
                ax.pcolormesh(R,P,streams,**kargs)
        else:
            if self.check:
                raise NotImplementedError("For now, we do not compute streamlines in the (R,z) plane")

def is_averageSafe(sigma0,sigmaSlope,plot=False):
    init = InitParamNonos() # initialize the major parameters
    fieldon = FieldNonos(init, field='RHO', on=0) # fieldon object with the density field at on=0
    datarz=np.mean(fieldon.data,axis=1) # azimuthally-averaged density field
    error=(sigma0*pow(fieldon.xmed,-sigmaSlope)-np.mean(datarz, axis=1)*next(item for item in [fieldon.z.max()-fieldon.z.min(),1.0] if item!=0))/(sigma0*pow(fieldon.xmed,-sigmaSlope)) # comparison between Sigma(R) profile and integral of rho(R,z) between zmin and zmax
    if any(100*abs(error)>3):
        print("With a maximum of %.1f percents of error, the averaging procedure may not be safe.\nzmax/h is probably too small.\nUse rather average=False (-noavr) or increase zmin/zmax."%np.max(100*abs(error)))
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

def print_warn(message):
    """
    adapted from idefix_cli (cmt robert)
    https://github.com/neutrinoceros/idefix_cli
    """
    rprint(f"[bold red]Warning |[/] {message}", file=sys.stderr)

def print_err(message):
    """
    adapted from idefix_cli (cmt robert)
    https://github.com/neutrinoceros/idefix_cli
    """
    rprint(f"[bold white on red]Error |[/] {message}", file=sys.stderr)

# process function for parallisation purpose with progress bar
# counterParallel = Value('i', 0) # initialization of a counter
def process_field(on, init, profile, field, mid, cart, avr, diff, log, corotate, streamlines, stype, srmin, srmax, nstream, vmin, vmax, ft, cmap, isPlanet, pbar, parallel, directory):
    ploton=PlotNonos(init, field=field, on=on, diff=diff, log=log, corotate=corotate, isPlanet=isPlanet, directory=directory, check=False)
    try:
        if streamlines:
            streamon=StreamNonos(init, field=field, on=on, directory=directory, check=False)
            vx1on = FieldNonos(init, field='VX1', on=on, diff=False, log=False, corotate=corotate, isPlanet=isPlanet, directory=directory, check=False)
            vx2on = FieldNonos(init, field='VX2', on=on, diff=False, log=False, corotate=corotate, isPlanet=isPlanet, directory=directory, check=False)
    except FileNotFoundError as exc:
        print_err(exc)
        return 1

    if (not cart and not mid):
        print_warn('plot not optimized for now in the (R,z) plane in polar.\nCheck in cartesian coordinates to be sure')
        fig = plt.figure(figsize=(9,8))
        ax = fig.add_subplot(111, polar=True)
    else:
        fig, ax=plt.subplots(figsize=(9,8))#, sharex=True, sharey=True)
    plt.subplots_adjust(left=0.1, right=0.87, top=0.95, bottom=0.1)
    plt.ioff()

    # plot the field
    if profile=="2d":
        try:
            ploton.plot(ax, vmin=vmin, vmax=vmax, midplane=mid, cartesian=cart, average=avr, fontsize=ft, cmap=cmap)
        except IndexError as exc:
            print_err(exc)
            return 1
        if streamlines:
            vr = vx1on.data
            vphi = vx2on.data
            if isPlanet:
                vphi -= vx2on.omegaplanet[on]*vx2on.xmed[:,None,None]
            if stype=="lic":
                streams=streamon.get_lic_streams(vr,vphi)
                streamon.plot_lic(ax,streams,cartesian=cart, cmap='gray', alpha=0.3)
            elif stype=="random":
                streams=streamon.get_random_streams(vr,vphi,xmin=srmin,xmax=srmax, n=nstream)
                streamon.plot_streams(ax,streams,cartesian=cart,color='k', linewidth=2, alpha=0.5)
            elif stype=="fixed":
                streams=streamon.get_fixed_streams(vr,vphi,xmin=srmin,xmax=srmax, n=nstream)
                streamon.plot_streams(ax,streams,cartesian=cart,color='k', linewidth=2, alpha=0.5)

        if mid:
            plt.savefig("Rphi_%s_diff%slog%s_c%s%04d.png"%(field,diff,log,cart,on))
        else:
            plt.savefig("Rz_%s_diff%slog%s_c%s%04d.png"%(field,diff,log,cart,on))

    # plot the 1D profile
    if profile=="1d":
        ploton.axiplot(ax, vmin=vmin, vmax=vmax, average=avr, fontsize=ft)
        plt.savefig("axi_%s_diff%slog%s%04d.png"%(field,diff,log,on))

    plt.close()

    # if pbar:
    #     if parallel:
    #         global counterParallel
    #         printProgressBar(counterParallel.value, len(config['onarray'])-1, prefix = 'Progress:', suffix = 'Complete', length = 50) # progress bar when parallelization is included
    #         with counterParallel.get_lock():
    #             counterParallel.value += 1  # incrementation of the counter
    #     else:
    #         printProgressBar(on-config['onarray'][0], len(config['onarray'])-1, prefix = 'Progress:', suffix = 'Complete', length = 50)

def main(argv: Optional[List[str]] = None, show=True) -> int:
    # read the .toml file
    try:
        analysis = AnalysisNonos()
    except FileNotFoundError as exc:
        print_err(exc)
        return 1

    pconfig=analysis.config

    parser = argparse.ArgumentParser(prog='nonos',
                                     usage='%(prog)s -mod d/f [options]',
                                     description='Analysis tool for idefix/pluto/fargo3d simulations (in polar coordinates)',
                                     )
    # analysis = AnalysisNonos(directory=args.dir)
    parser.add_argument(
        '-l',
        action="store_true",
        help="local mode",
        )
    parser.add_argument(
        '-info',
        action="store_true",
        help="give the default parameters in the config.toml file.",
        )
    parser.add_argument(
        '-dir',
        type=str,
        default=pconfig['dir'],
        help="where .vtk files and the inifile are stored ('.' by default).",
        )
    parser.add_argument(
        '-mod',
        type=str,
        choices=["","d","f"],
        default="",
        help="display/film ('' home page by default).",
        )
    parser.add_argument(
        '-on',
        type=int,
        default=pconfig['onStart'],
        help="if -mod d, we plot the field of the data.on.vtk file (1 by default).",
        )
    parser.add_argument(
        '-f',
        type=str.lower,
        default=pconfig['field'],
        help="field (for now RHO, VX1 and VX2 in 2D, + VX3 in 3D, RHO by default).",
        )
    parser.add_argument(
        '-vmin',
        type=float,
        default=None,
        help="minimum value for the data (-0.5 by default or calculated.",
        )
    parser.add_argument(
        '-vmax',
        type=float,
        default=None,
        help="maximum value for the data (0.5 by default or calculated.",
        )
    parser.add_argument(
        '-onend',
        type=int,
        default=pconfig['onEnd'],
        help="if -mod f and -partial (15 by default).",
        )
    parser.add_argument(
        '-diff',
        action="store_true",
        help="plot the relative perturbation of the field f, i.e. (f-f0)/f0 (false by default).",
        )
    parser.add_argument(
        '-log',
        action="store_true",
        help="plot the log of the field f, i.e. log(f) (false by default).",
        )
    parser.add_argument(
        '-cor',
        action="store_true",
        help="does the grid corotate? For now, works in pair with -isp (false by default).",
        )
    parser.add_argument(
        '-s',
        action="store_true",
        help="do we compute streamlines? (false by default)",
        )
    parser.add_argument(
        '-stype',
        type=str,
        choices=["random","fixed","lic"],
        default=pconfig['streamtype'],
        help="do we compute random, fixed streams, or do we use line integral convolution? (random by default)",
        )
    parser.add_argument(
        '-srmin',
        type=float,
        default=pconfig['rminStream'],
        help="minimum radius for streamlines computation (0.7 by default).",
        )
    parser.add_argument(
        '-srmax',
        type=float,
        default=pconfig['rmaxStream'],
        help="maximum radius for streamlines computation (1.3 by default).",
        )
    parser.add_argument(
        '-sn',
        type=int,
        default=pconfig['nstream'],
        help="number of streamlines (50 by default).",
        )
    parser.add_argument(
        '-isp',
        action="store_true",
        help="is there a planet in the grid ? (false by default)",
        )

    groupmid = parser.add_mutually_exclusive_group()
    groupmid.add_argument(
        '-mid',
        action="store_true",
        default=pconfig['midplane'],
        help="2D plot in the (R-phi) plane (true by default).",
        )
    groupmid.add_argument(
        '-rz',
        action="store_true",
        default=False,
        help="2D plot in the (R-z) plane (false by default).",
        )
    groupcart = parser.add_mutually_exclusive_group()
    groupcart.add_argument(
        '-cart',
        action="store_true",
        default=pconfig['cartesian'],
        help="2D plot in cartesian coordinates (true by default).",
        )
    groupcart.add_argument(
        '-pol',
        action="store_true",
        default=False,
        help="2D plot in polar coordinates (false by default).",
        )
    groupavr = parser.add_mutually_exclusive_group()
    groupavr.add_argument(
        '-avr',
        action="store_true",
        default=pconfig['average'],
        help="do we average in the 3rd dimension, i.e. vertically when -mid and azimuthally when -rz (true by default).",
        )
    groupavr.add_argument(
        '-noavr',
        action="store_true",
        default=False,
        help="do we consider a specific plane, i.e. at the midplane (z=0) when -mid and at phi=0 when -rz (false by default).",
        )

    parser.add_argument(
        '-p',
        type=str,
        choices=["2d","1d"],
        default=pconfig['profile'],
        help="1D axisymmetric radial profile or 2D field (2d by default).",
        )
    parser.add_argument(
        '-ft',
        type=float,
        default=pconfig['fontsize'],
        help="fontsize in the graph (11 by default).",
        )
    parser.add_argument(
        '-cmap',
        type=str,
        default=pconfig['cmap'],
        help="choice of colormap for the -p 2d maps (RdYlBu_r by default).",
        )
    parser.add_argument(
        '-partial',
        action="store_true",
        default=False,
        help="if -mod f, partial movie between -on and -onend (false by default).",
        )
    parser.add_argument(
        '-pbar',
        action="store_true",
        help="do we display the progress bar when -mod f? (false by default)",
        )
    parser.add_argument(
        '-multi',
        action="store_true",
        help="load and save figures in parallel when -mod f (false by default).",
        )
    parser.add_argument(
        '-cpu',
        type=int,
        default=pconfig['nbcpu'],
        help="number of cpus if -multi (4 by default).",
        )

    args = parser.parse_args(argv)

    if args.l:
        rprint("[bold white]Local mode")
        if len(glob.glob1("","config.toml"))!=1:
            pathconfig = pkg_resources.resource_filename("nonos", "config.toml")
            copyfile(pathconfig, "config.toml")
            print_warn("config.toml file copied in working directory.\nYou can now open it and choose the parameters")
            return 0
        try:
            init = InitParamNonos(directory=args.dir, directory_of_script="", info=args.info)
        except (FileNotFoundError,RuntimeError,ValueError) as exc:
            print_err(exc)
            return 1
        args.dir=init.config["dir"]
        args.mod=init.config["mode"]
        args.on=init.config["onStart"]
        args.f=init.config["field"]
        args.onend=init.config["onEnd"]
        args.diff=init.config["diff"]
        args.log=init.config["log"]
        if args.diff:
            args.vmin=init.config["vmin"]
            args.vmax=init.config["vmax"]
        else:
            args.vmin=None
            args.vmax=None
        args.cor=init.config["corotate"]
        args.s=init.config["streamlines"]
        args.stype=init.config["streamtype"]
        args.srmin=init.config["rminStream"]
        args.srmax=init.config["rmaxStream"]
        args.sn=init.config["nstream"]
        args.isp=init.config["isPlanet"]
        args.mid=init.config["midplane"]
        args.rz=not args.mid
        args.cart=init.config["cartesian"]
        args.pol=not args.cart
        args.avr=init.config["average"]
        args.noavr=not args.avr
        args.p=init.config["profile"]
        args.ft=init.config["fontsize"]
        args.cmap=init.config["cmap"]
        args.partial=not init.config["fullfilm"]
        args.pbar=init.config["progressBar"]
        args.multi=init.config["parallel"]
        args.cpu=init.config["nbcpu"]
    else:
        try:
            init = InitParamNonos(directory=args.dir, info=args.info)
        except (FileNotFoundError,RuntimeError,ValueError) as exc:
            print_err(exc)
            return 1

    n_file=init.n_file
    diran=init.directory

    args.f=args.f.upper()
    if args.rz:
        args.mid=False
    if args.pol:
        args.cart=False
    if args.noavr:
        args.avr=False

    if(not(args.mid) and args.s):
        print_err("For now, we do not compute streamlines in the (R,z) plane")
        return 1

    if(not(args.isp) and args.cor):
        print_warn("We don't rotate the grid if there is no planet for now.\nomegagrid = 0.")

    if(args.s and args.stype=='lic'):
        print_warn("TODO: check what is the length argument in StreamNonos().get_lic_streams ?")

    # mode for just displaying a field for a given output number
    if args.mod=="d":
        if (args.pol and args.rz):
            print_warn('plot not optimized for now in the (R,z) plane in polar.\nCheck in cartesian coordinates to be sure')
            fig = plt.figure(figsize=(9,8))
            ax = fig.add_subplot(111, polar=True)
        else:
            fig, ax=plt.subplots(figsize=(9,8))#, sharex=True, sharey=True)
        plt.ioff()
        # print("on = ", args.on)
        # loading the field

        try:
            ploton = PlotNonos(init, field=args.f, on=args.on, diff=args.diff, log=args.log, corotate=args.cor, isPlanet=args.isp, directory=diran, check=False)
            if args.s:
                streamon=StreamNonos(init, field=args.f, on=args.on, directory=diran, check=False)
                vx1on = FieldNonos(init, field='VX1', on=args.on, diff=False, log=False, corotate=args.cor, isPlanet=args.isp, directory=diran, check=False)
                vx2on = FieldNonos(init, field='VX2', on=args.on, diff=False, log=False, corotate=args.cor, isPlanet=args.isp, directory=diran, check=False)
        except FileNotFoundError as exc:
            print_err(exc)
            return 1

        # plot the field
        if args.p=="2d":
            try:
                ploton.plot(ax, vmin=args.vmin, vmax=args.vmax, midplane=args.mid, cartesian=args.cart, average=args.avr, fontsize=args.ft, cmap=args.cmap)
            except IndexError as exc:
                print_err(exc)
                return 1
            if args.s:
                vr = vx1on.data
                vphi = vx2on.data
                if args.isp:
                    vphi -= vx2on.omegaplanet[args.on]*vx2on.xmed[:,None,None]
                if args.stype=="lic":
                    streams=streamon.get_lic_streams(vr,vphi)
                    streamon.plot_lic(ax,streams,cartesian=args.cart, cmap='gray', alpha=0.3)
                elif args.stype=="random":
                    streams=streamon.get_random_streams(vr,vphi,xmin=args.srmin,xmax=args.srmax, n=args.sn)
                    streamon.plot_streams(ax,streams,cartesian=args.cart,color='k', linewidth=2, alpha=0.5)
                elif args.stype=="fixed":
                    streams=streamon.get_fixed_streams(vr,vphi,xmin=args.srmin,xmax=args.srmax, n=args.sn)
                    streamon.plot_streams(ax,streams,cartesian=args.cart,color='k', linewidth=2, alpha=0.5)

        # plot the 1D profile
        if args.p=="1d":
            ploton.axiplot(ax, vmin=args.vmin, vmax=args.vmax, average=args.avr, fontsize=args.ft)

        if show:
            plt.show()
        else:
            plt.close()

    # mode for creating a movie of the temporal evolution of a given field
    elif args.mod=="f":
        # do we compute the full movie or a partial movie given by "on"
        if args.partial:
            init.config['onarray']=np.arange(args.on,args.onend+1)
        else:
            init.config['onarray']=range(n_file)

        # calculation of the min/max
        if args.diff:
            if args.vmin is None:
                args.vmin=init.config['vmin']
            if args.vmax is None:
                args.vmax=init.config['vmax']
        # In that case we choose a file in the middle (len(onarray)//2) and compute the MIN/MAX
        else:
            try:
                fieldon = FieldNonos(init, field=args.f, on=init.config['onarray'][len(init.config['onarray'])//2], directory=diran, diff=False, check=False)
            except FileNotFoundError as exc:
                print_err(exc)
                return 1
            if args.p=="2d":
                if args.vmin is None:
                    args.vmin=fieldon.data.min()
                if args.vmax is None:
                    args.vmax=fieldon.data.max()
            elif args.p=="1d":
                if args.vmin is None:
                    args.vmin=(np.mean(np.mean(fieldon.data,axis=1),axis=1)).min()
                if args.vmax is None:
                    args.vmax=(np.mean(np.mean(fieldon.data,axis=1),axis=1)).max()

        # call of the process_field function, whether it be in parallel or not
        # if args.pbar:
        #     printProgressBar(0, len(pconfig['onarray'])-1, prefix = 'Progress:', suffix = 'Complete', length = 50) # progress bar when parallelization is included
        # if args.multi:
        #     # determines the minimum between nbcpu and the nb max of cpus in the user's system
        #     nbcpuReal = min((int(args.cpu),os.cpu_count()))
        #     pool = Pool(nbcpuReal)   # Create a multiprocessing Pool with a security on the number of cpus
        #     pool.map(functools.partial(process_field, profile=args.p, field=args.f, mid=args.mid, cart=args.cart, avr=args.avr, diff=args.diff, log=args.log, corotate=args.cor, streamlines=args.s, stype=args.stype, srmin=args.srmin, srmax=args.srmax, nstream=args.sn, config=pconfig, vmin=args.vmin, vmax=args.vmax, ft=args.ft, cmap=args.cmap, isPlanet=args.isp, pbar=args.pbar, parallel=args.multi, directory=diran), pconfig['onarray'])
        #     tpara=time.time()-start
        #     print("time in parallel : %f" %tpara)
        start=time.time()
        if args.multi:
            # determines the minimum between nbcpu and the nb max of cpus in the user's system
            nbcpuReal = min((int(args.cpu),os.cpu_count()))
            if args.pbar:
                with Pool(nbcpuReal) as pool:   # Create a multiprocessing Pool with a security on the number of cpus
                    list(track(pool.imap(functools.partial(process_field, init=init, profile=args.p, field=args.f, mid=args.mid, cart=args.cart, avr=args.avr, diff=args.diff, log=args.log, corotate=args.cor, streamlines=args.s, stype=args.stype, srmin=args.srmin, srmax=args.srmax, nstream=args.sn, vmin=args.vmin, vmax=args.vmax, ft=args.ft, cmap=args.cmap, isPlanet=args.isp, pbar=args.pbar, parallel=args.multi, directory=diran), init.config['onarray']), total=len(init.config['onarray'])))
            else:
                pool = Pool(nbcpuReal)   # Create a multiprocessing Pool with a security on the number of cpus
                pool.map(functools.partial(process_field, init=init, profile=args.p, field=args.f, mid=args.mid, cart=args.cart, avr=args.avr, diff=args.diff, log=args.log, corotate=args.cor, streamlines=args.s, stype=args.stype, srmin=args.srmin, srmax=args.srmax, nstream=args.sn, vmin=args.vmin, vmax=args.vmax, ft=args.ft, cmap=args.cmap, isPlanet=args.isp, pbar=args.pbar, parallel=args.multi, directory=diran), init.config['onarray'])
            tpara=time.time()-start
            print("time in parallel : %f" %tpara)
        else:
            if args.pbar:
                list(map(functools.partial(process_field, init=init, profile=args.p, field=args.f, mid=args.mid, cart=args.cart, avr=args.avr, diff=args.diff, log=args.log, corotate=args.cor, streamlines=args.s, stype=args.stype, srmin=args.srmin, srmax=args.srmax, nstream=args.sn, vmin=args.vmin, vmax=args.vmax, ft=args.ft, cmap=args.cmap, isPlanet=args.isp, pbar=args.pbar, parallel=args.multi, directory=diran), track(init.config['onarray'])))
            else:
                list(map(functools.partial(process_field, init=init, profile=args.p, field=args.f, mid=args.mid, cart=args.cart, avr=args.avr, diff=args.diff, log=args.log, corotate=args.cor, streamlines=args.s, stype=args.stype, srmin=args.srmin, srmax=args.srmax, nstream=args.sn, vmin=args.vmin, vmax=args.vmax, ft=args.ft, cmap=args.cmap, isPlanet=args.isp, pbar=args.pbar, parallel=args.multi, directory=diran), init.config['onarray']))
            tserie=time.time()-start
            print("time in serie : %f" %tserie)

    elif args.mod=="":
        print("""
                                                             `!)}$$$$})!`
         `,!>|))|!,                                        :}&#&$}{{}$&#&};`
      ~)$&&$$}}$$&#$).                                   '}#&}|~'.```.'!($#$+
   `=$&$(^,..``.'~=$&&=                                `|&#}!'`         `'?$#}.
  !$&}:.`         `.!$#$'                             :$#$^'`       ``     .$#$`
 ^&&+.              `'(&$!                          `)&&),`                 !##!
`$#^      `.   `     `={$&{`                       ^$&$(!'  ``     `,.  ``  !##!
,#$ ``                 .>}&${?!:'`   ```..'',:!+|{$&${:`   `'`             `}&}`
`$$`                       ,;|{}$$$&&&&&$$$$}}()<!,`   '`                 `}&}`
 +&}`   `   |:|.\    |:|            `.```                                  :$$!
  !$$'`!}|  |:|\.\   |:|      __                      __       ___       .{#$
   '$&})$:  |:| \.\  |:|   /./  \.\   |:|.\  |:|   /./  \.\   |:|  \.\  '$$#}
    `}#&;   |:|  \.\ |:|  |:|    |:|  |:|\.\ |:|  |:|    |:|  |:|___     :$)&$`
    `{&!    |:|   \.\|:|  |:|    |:|  |:| \.\|:|  |:|    |:|       |:|    :!{&}`
   :$$,     |:|    \.|:|   \.\__/./   |:|  \.|:|   \.\__/./   \.\__|:|     `:}#}`
  ^&$.                                                                       .$#^
  +&$.                 ``'^)}$$$$$({}}}$$$$$$$$$$}}(|>!.`~:,.                 }#)
 '&#|                 ,|$##$>'`                `'~!)$##$$$)?^,`           ` :&&:
 ,&#}`  ``       .` `:{$&}:                          ~}&$)^^+=^`  `` ..  .|&#)
  |&#$:```   `` '::!}$&},                              !$&$|++^^:,:~',!!($#&^
   ,}&#${^~,,,:!|}$&&(.                                  ^$#$}{)|?|)(}$&#$?`
     :{&##$$$$$&##$).                                      ~($&#&&##&$}=,
       `:|}$$$$}):`                                           `',,,.`
              """)
        print("Analysis tool for idefix/pluto/fargo3d simulations (in polar coordinates)")

    return 0
