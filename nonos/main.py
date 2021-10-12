#!/usr/bin/env python
"""
Analysis tool for idefix/pluto/fargo3d simulations (in polar coordinates).
"""
# adapted from pbenitez-llambay, gwafflard-fernandez, cmt robert & glesur

import argparse
import functools
import glob
import logging
import os
import re
import time
from collections import ChainMap
from copy import copy
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cblind as cb
import inifix
import matplotlib.pyplot as plt
import numpy as np
import pkg_resources
import pytomlpp as toml
from inifix.format import iniformat
from licplot import lic_internal
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rich.logging import RichHandler
from scipy.interpolate import griddata
from skimage import exposure
from skimage.util import random_noise

from nonos.__version__ import __version__
from nonos.config import DEFAULTS
from nonos.geometry import (
    GEOM_TRANSFORMS,
    get_keys_from_geomtransforms,
    meshgrid_from_plane,
)
from nonos.logging import parse_verbose_level, print_err, print_warn
from nonos.parsing import (
    is_set,
    parse_image_format,
    parse_output_number_range,
    parse_range,
    parse_vmin_vmax,
    range_converter,
)
from nonos.styling import set_mpl_style

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


def readVTK(filename, *, geometry="unknown", cell="edges", computedata=True):
    """
    Adapted from Geoffroy Lesur
    Function that reads a vtk file in polar coordinates
    """
    nfound = len(glob.glob(filename))
    if nfound != 1:
        raise FileNotFoundError("In readVTK: %s not found." % filename)

    fid = open(filename, "rb")

    # define our datastructure
    V = DataStructure()

    # raw data which will be read from the file
    V.data = {}

    # initialize geometry
    if geometry not in ("unknown", "cartesian", "polar", "spherical"):
        raise ValueError(f"Received unknown geometry: '{geometry}'.")
    V.geometry = geometry

    # datatype we read
    dt = np.dtype(">f")  # Big endian single precision floats
    dint = np.dtype(">i4")  # Big endian integer

    s = fid.readline()  # VTK DataFile Version x.x
    s = fid.readline()  # Comments

    s = fid.readline()  # BINARY
    s = fid.readline()  # DATASET RECTILINEAR_GRID or STRUCTURED_GRID
    slist = s.split()

    s = fid.readline()  # DIMENSIONS NX NY NZ
    slist = s.split()
    entry = str(slist[0], "utf-8")
    if entry == "FIELD":
        nfield = int(slist[2])
        for _field in range(nfield):
            s = fid.readline()
            slist = s.split()
            entry = str(slist[0], "utf-8")
            if entry == "TIME":
                V.t = np.fromfile(fid, dt, 1)
            elif entry == "GEOMETRY":
                g = np.fromfile(fid, dint, 1)
                if g == 0:
                    thisgeometry = "cartesian"
                elif g == 1:
                    thisgeometry = "polar"
                elif g == 2:
                    thisgeometry = "spherical"
                else:
                    raise ValueError(
                        f"Unknown value for GEOMETRY flag ('{g}') was found in the VTK file."
                    )

                if V.geometry != "unknown":
                    # We already have a proposed geometry, check that what is read from the file matches
                    if thisgeometry != V.geometry:
                        raise ValueError(
                            f"geometry argument ('{V.geometry}') is inconsistent with GEOMETRY flag from the VTK file ('{thisgeometry}')"
                        )
                V.geometry = thisgeometry
            else:
                raise ValueError(f"Received unknown field: '{entry}'.")

            s = fid.readline()  # extra linefeed

        # finished reading the field entry
        # read next line
        s = fid.readline()  # DIMENSIONS...

    if V.geometry == "unknown":
        raise RuntimeError(
            "Geometry couldn't be determined from data. "
            "Try to set the geometry keyword argument explicitely."
        )

    slist = s.split()  # DIMENSIONS....
    V.n1 = int(slist[1])
    V.n2 = int(slist[2])
    V.n3 = int(slist[3])

    if V.geometry == "cartesian":
        # CARTESIAN geometry
        s = fid.readline()  # X_COORDINATES NX float
        inipos = (
            fid.tell()
        )  # we store the file pointer position before computing points
        logging.debug("loading the X-grid cells: %d" % V.n1)
        x = np.memmap(
            fid, mode="r", dtype=dt, offset=inipos, shape=V.n1
        )  # some smart memory efficient way to store the array
        newpos = (
            np.float32().nbytes * 1 * V.n1 + inipos
        )  # we calculate the offset that we would expect normally with a np.fromfile
        fid.seek(newpos, os.SEEK_SET)  # we set the file pointer position to this offset
        s = fid.readline()  # Extra line feed added by idefix

        s = fid.readline()  # Y_COORDINATES NY float
        inipos = (
            fid.tell()
        )  # we store the file pointer position before computing points
        logging.debug("loading the Y-grid cells: %d" % V.n2)
        y = np.memmap(
            fid, mode="r", dtype=dt, offset=inipos, shape=V.n2
        )  # some smart memory efficient way to store the array
        newpos = (
            np.float32().nbytes * 1 * V.n2 + inipos
        )  # we calculate the offset that we would expect normally with a np.fromfile
        fid.seek(newpos, os.SEEK_SET)  # we set the file pointer position to this offset
        s = fid.readline()  # Extra line feed added by idefix

        s = fid.readline()  # Z_COORDINATES NZ float
        inipos = (
            fid.tell()
        )  # we store the file pointer position before computing points
        logging.debug("loading the Z-grid cells: %d" % V.n3)
        z = np.memmap(
            fid, mode="r", dtype=dt, offset=inipos, shape=V.n3
        )  # some smart memory efficient way to store the array
        newpos = (
            np.float32().nbytes * 1 * V.n3 + inipos
        )  # we calculate the offset that we would expect normally with a np.fromfile
        fid.seek(newpos, os.SEEK_SET)  # we set the file pointer position to this offset
        s = fid.readline()  # Extra line feed added by idefix

        s = fid.readline()  # POINT_DATA NXNYNZ

        slist = s.split()
        point_type = str(slist[0], "utf-8")
        npoints = int(slist[1])
        s = fid.readline()  # EXTRA LINE FEED

        if point_type == "CELL_DATA" or cell == "centers":
            # The file contains face coordinates, so we extrapolate to get the cell center coordinates.
            if V.n1 > 1:
                V.n1 = V.n1 - 1
                V.x1 = 0.5 * (x[1:] + x[:-1])
            else:
                V.x1 = x
            if V.n2 > 1:
                V.n2 = V.n2 - 1
                V.x2 = 0.5 * (y[1:] + y[:-1])
            else:
                V.x2 = y
            if V.n3 > 1:
                V.n3 = V.n3 - 1
                V.x3 = 0.5 * (z[1:] + z[:-1])
            else:
                V.x3 = z
        elif point_type == "POINT_DATA" or cell == "edges":
            V.x1 = x
            V.x2 = y
            V.x3 = z
            if V.n1 > 1:
                V.n1 = V.n1 - 1
                V.x1 = x
            else:
                V.x1 = x
            if V.n2 > 1:
                V.n2 = V.n2 - 1
                V.x2 = y
            else:
                V.x2 = y
            if V.n3 > 1:
                V.n3 = V.n3 - 1
                V.x3 = z
            else:
                V.x3 = z

        if V.n1 * V.n2 * V.n3 != npoints:
            raise ValueError(
                "In readVTK: Grid size (%d) incompatible with number of points (%d) in the data set"
                % (V.n1 * V.n2 * V.n3, npoints)
            )

    else:
        # POLAR or SPHERICAL coordinates
        if V.n3 == 1:
            is2d = 1
        else:
            is2d = 0

        s = fid.readline()  # POINTS NXNYNZ float
        slist = s.split()
        npoints = int(slist[1])

        inipos = (
            fid.tell()
        )  # we store the file pointer position before computing points
        # print(inipos)
        # points = np.fromfile(fid, dt, 3 * npoints)
        logging.debug("loading the grid cells: (%d,%d,%d)." % (V.n1, V.n2, V.n3))
        points = np.memmap(
            fid, mode="r", dtype=dt, offset=inipos, shape=3 * npoints
        )  # some smart memory efficient way to store the array
        # print(fid.tell())
        newpos = (
            np.float32().nbytes * 3 * npoints + inipos
        )  # we calculate the offset that we would expect normally with a np.fromfile
        fid.seek(newpos, os.SEEK_SET)  # we set the file pointer position to this offset
        # print(fid.tell())
        s = fid.readline()  # EXTRA LINE FEED

        # V.points=points
        if V.n1 * V.n2 * V.n3 != npoints:
            raise ValueError(
                "In readVTK: Grid size (%d) incompatible with number of points (%d) in the data set"
                % (V.n1 * V.n2 * V.n3, npoints)
            )

        # Reconstruct the polar coordinate system
        x1d = points[::3]
        y1d = points[1::3]
        z1d = points[2::3]

        xcart = np.transpose(x1d.reshape(V.n3, V.n2, V.n1))
        ycart = np.transpose(y1d.reshape(V.n3, V.n2, V.n1))
        zcart = np.transpose(z1d.reshape(V.n3, V.n2, V.n1))

        # Reconstruct the polar coordinate system
        if V.geometry == "polar":

            r = np.sqrt(xcart[:, 0, 0] ** 2 + ycart[:, 0, 0] ** 2)
            theta = np.unwrap(np.arctan2(ycart[0, :, 0], xcart[0, :, 0]))
            z = zcart[0, 0, :]

            s = fid.readline()  # CELL_DATA (NX-1)(NY-1)(NZ-1)
            slist = s.split()
            data_type = str(slist[0], "utf-8")
            if data_type != "CELL_DATA":
                fid.close()
                raise ValueError(
                    "In readVTK: this routine expect 'CELL DATA' as produced by PLUTO, not '%s'."
                    % data_type
                )
            s = fid.readline()  # Line feed

            if cell == "edges":
                if V.n1 > 1:
                    V.n1 = V.n1 - 1
                    V.x1 = r
                else:
                    V.x1 = r
                if V.n2 > 1:
                    V.n2 = V.n2 - 1
                    V.x2 = theta
                else:
                    V.x2 = theta
                if V.n3 > 1:
                    V.n3 = V.n3 - 1
                    V.x3 = z
                else:
                    V.x3 = z

            # Perform averaging on coordinate system to get cell centers
            # The file contains face coordinates, so we extrapolate to get the cell center coordinates.
            elif cell == "centers":
                if V.n1 > 1:
                    V.n1 = V.n1 - 1
                    V.x1 = 0.5 * (r[1:] + r[:-1])
                else:
                    V.x1 = r
                if V.n2 > 1:
                    V.n2 = V.n2 - 1
                    V.x2 = (0.5 * (theta[1:] + theta[:-1]) + np.pi) % (
                        2.0 * np.pi
                    ) - np.pi
                else:
                    V.x2 = theta
                if V.n3 > 1:
                    V.n3 = V.n3 - 1
                    V.x3 = 0.5 * (z[1:] + z[:-1])
                else:
                    V.x3 = z

        # Reconstruct the spherical coordinate system
        if V.geometry == "spherical":
            if is2d:
                r = np.sqrt(xcart[:, 0, 0] ** 2 + ycart[:, 0, 0] ** 2)
                phi = np.unwrap(
                    np.arctan2(zcart[0, V.n2 // 2, :], xcart[0, V.n2 // 2, :])
                )
                theta = np.arccos(
                    ycart[0, :, 0] / np.sqrt(xcart[0, :, 0] ** 2 + ycart[0, :, 0] ** 2)
                )
            else:
                r = np.sqrt(
                    xcart[:, 0, 0] ** 2 + ycart[:, 0, 0] ** 2 + zcart[:, 0, 0] ** 2
                )
                phi = np.unwrap(
                    np.arctan2(
                        ycart[V.n1 // 2, V.n2 // 2, :], xcart[V.n1 // 2, V.n2 // 2, :]
                    )
                )
                theta = np.arccos(
                    zcart[0, :, 0]
                    / np.sqrt(
                        xcart[0, :, 0] ** 2 + ycart[0, :, 0] ** 2 + zcart[0, :, 0] ** 2
                    )
                )

            s = fid.readline()  # CELL_DATA (NX-1)(NY-1)(NZ-1)
            slist = s.split()
            data_type = str(slist[0], "utf-8")
            if data_type != "CELL_DATA":
                fid.close()
                raise ValueError(
                    "In readVTK: this routine expect 'CELL DATA' as produced by PLUTO, not '%s'."
                    % data_type
                )
            s = fid.readline()  # Line feed

            if cell == "edges":
                if V.n1 > 1:
                    V.n1 = V.n1 - 1
                    V.x1 = r
                else:
                    V.x1 = r
                if V.n2 > 1:
                    V.n2 = V.n2 - 1
                    V.x2 = theta
                else:
                    V.x2 = theta
                if V.n3 > 1:
                    V.n3 = V.n3 - 1
                    V.x3 = phi
                else:
                    V.x3 = phi

            # Perform averaging on coordinate system to get cell centers
            # The file contains face coordinates, so we extrapolate to get the cell center coordinates.
            elif cell == "centers":
                if V.n1 > 1:
                    V.n1 = V.n1 - 1
                    V.x1 = 0.5 * (r[1:] + r[:-1])
                else:
                    V.x1 = r
                if V.n2 > 1:
                    V.n2 = V.n2 - 1
                    V.x2 = 0.5 * (theta[1:] + theta[:-1])
                else:
                    V.x2 = theta
                if V.n3 > 1:
                    V.n3 = V.n3 - 1
                    V.x3 = 0.5 * (phi[1:] + phi[:-1])
                else:
                    V.x3 = phi

    if computedata:
        logging.debug("loading the data arrays:")
        while 1:
            s = (
                fid.readline()
            )  # SCALARS/VECTORS name data_type (ex: SCALARS imagedata unsigned_char)
            # print repr(s)
            if len(s) < 2:  # leave if end of file
                break
            slist = s.split()
            datatype = str(slist[0], "utf-8")
            varname = str(slist[1], "utf-8")
            if datatype == "SCALARS":
                fid.readline()  # LOOKUP TABLE

                inipos = (
                    fid.tell()
                )  # we store the file pointer position before computing points
                # array = np.fromfile(fid, dt, V.n1 * V.n2 * V.n3).reshape(V.n3, V.n2, V.n1)
                array = np.memmap(
                    fid, mode="r", dtype=dt, offset=inipos, shape=V.n1 * V.n2 * V.n3
                ).reshape(
                    V.n3, V.n2, V.n1
                )  # some smart memory efficient way to store the array
                newpos = (
                    np.float32().nbytes * V.n1 * V.n2 * V.n3 + inipos
                )  # we calculate the offset that we would expect normally with a np.fromfile
                fid.seek(
                    newpos, os.SEEK_SET
                )  # we set the file pointer position to this offset

                V.data[varname] = np.transpose(array)
            elif datatype == "VECTORS":
                inipos = (
                    fid.tell()
                )  # we store the file pointer position before computing points
                Q = np.memmap(
                    fid, mode="r", dtype=dt, offset=inipos, shape=V.n1 * V.n2 * V.n3
                )  # some smart memory efficient way to store the array
                # Q = np.fromfile(fid, dt, 3 * V.n1 * V.n2 * V.n3)
                newpos = (
                    np.float32().nbytes * V.n1 * V.n2 * V.n3 + inipos
                )  # we calculate the offset that we would expect normally with a np.fromfile
                fid.seek(
                    newpos, os.SEEK_SET
                )  # we set the file pointer position to this offset

                V.data[varname + "_X"] = np.transpose(Q[::3].reshape(V.n3, V.n2, V.n1))
                V.data[varname + "_Y"] = np.transpose(Q[1::3].reshape(V.n3, V.n2, V.n1))
                V.data[varname + "_Z"] = np.transpose(Q[2::3].reshape(V.n3, V.n2, V.n1))

            else:
                raise ValueError(
                    "In readVTK: Unknown datatype '%s', should be 'SCALARS' or 'VECTORS'"
                    % datatype
                )
                break

            logging.debug("field: %s" % varname)

            fid.readline()  # extra line feed
    fid.close()

    return V


# Former geometry-specific readers
def readVTKCart(filename, *, cell="edges", computedata=True):
    Warning(
        "the use of readVTKCart is discouraged. Use the generic readVTK function with geometry='cartesian'"
    )
    return readVTK(filename, geometry="cartesian", cell=cell, computedata=computedata)


# Read a vtk file
def readVTKPolar(filename, *, cell="edges", computedata=True):
    Warning(
        "the use of readVTKPolar is discouraged. Use the generic readVTK function with geometry='polar'"
    )
    return readVTK(filename, geometry="polar", cell=cell, computedata=computedata)


# Read a vtk file
def readVTKSpherical(filename, *, cell="edges", computedata=True):
    Warning(
        "the use of readVTKSpherical is discouraged. Use the generic readVTK function with geometry='spherical'"
    )
    return readVTK(filename, geometry="spherical", cell=cell, computedata=computedata)


class InitParamNonos:
    """
    Adapted from Pablo Benitez-Llambay
    Class for reading the simulation parameters.
    input: string -> name of the parfile, normally *.ini
    """

    def __init__(self, nonos_config=None, sim_paramfile=None, **kwargs):
        if nonos_config is None:
            nonos_config = copy(DEFAULTS)
        if diff := set(kwargs).difference(set(DEFAULTS)):
            raise TypeError(f"Received the following unsupported argument(s): {diff}")

        self.config = ChainMap(kwargs, nonos_config)
        self.paramfile = sim_paramfile

        lookup_table = {
            "idefix.ini": "idefix",
            "pluto.ini": "pluto",
            "variables.par": "fargo3d",
        }
        if self.paramfile is None:
            found = {
                paramfile: Path(self.config["datadir"]).joinpath(paramfile).is_file()
                for paramfile in lookup_table
            }
            nfound = sum(list(found.values()))
            if nfound == 0:
                raise FileNotFoundError(
                    "idefix.ini, pluto.ini or variables.par not found."
                )
            elif nfound > 1:
                raise RuntimeError("found more than one possible ini file.")
            self.paramfile = list(lookup_table.keys())[list(found.values()).index(True)]
        elif self.paramfile not in lookup_table:
            raise FileNotFoundError(
                "For now, impossible to choose your parameter file.\nBy default, the code searches idefix.ini, pluto.ini or variables.par."
            )

        self.code = lookup_table[self.paramfile]

    def load(self):
        self.iniconfig = inifix.load(
            os.path.join(self.config["datadir"], self.paramfile)
        )

        if self.code == "idefix":
            self.data_files = list(glob.glob1(self.config["datadir"], "data.*.vtk"))
            # self.h0 = self.iniconfig["Setup"]["h0"]
            if self.config["isPlanet"]:
                if Path(self.config["datadir"]).joinpath("planet0.dat").is_file():
                    with open(
                        os.path.join(self.config["datadir"], "planet0.dat")
                    ) as f1:
                        datafile = f1.readlines()
                        self.qpl = np.array(
                            [float(line.split()[7]) for line in datafile]
                        )
                        self.dpl = np.array(
                            [
                                np.sqrt(
                                    float(line.split()[1]) ** 2
                                    + float(line.split()[2]) ** 2
                                    + float(line.split()[3]) ** 2
                                )
                                for line in datafile
                            ]
                        )
                        self.xpl = np.array(
                            [float(line.split()[1]) for line in datafile]
                        )
                        self.ypl = np.array(
                            [float(line.split()[2]) for line in datafile]
                        )
                        self.tpl = np.array(
                            [float(line.split()[8]) for line in datafile]
                        )
                else:
                    self.qpl = np.array(
                        [self.iniconfig["Planet"]["qpl"] for _ in self.data_files]
                    )
                    self.dpl = np.array(
                        [self.iniconfig["Planet"]["dpl"] for _ in self.data_files]
                    )
                self.omegaplanet = np.sqrt(
                    (1.0 + self.qpl) / self.dpl / self.dpl / self.dpl
                )

            if self.config["corotate"]:
                self.vtk = self.iniconfig["Output"]["vtk"]
                if self.config["isPlanet"]:
                    self.omegagrid = self.omegaplanet
                else:
                    self.omegagrid = np.zeros(len(self.data_files))

        elif self.code == "pluto":
            self.data_files = list(glob.glob1(self.config["datadir"], "data.*.vtk"))
            # self.h0 = 0.05
            if self.config["isPlanet"]:
                self.qpl = np.full(
                    len(self.data_files),
                    self.iniconfig["Parameters"]["Mplanet"]
                    / self.iniconfig["Parameters"]["Mstar"],
                )
                print_warn(
                    "Initial distance not defined in pluto.ini.\nBy default, dpl=1.0 for the computation of omegaP\n"
                )
                self.dpl = np.ones(len(self.data_files))
                self.omegaplanet = np.sqrt(
                    (1.0 + self.qpl) / self.dpl / self.dpl / self.dpl
                )

            if self.config["corotate"]:
                self.vtk = self.iniconfig["Static Grid Output"]["vtk"][0]
                if self.config["isPlanet"]:
                    self.omegagrid = self.omegaplanet
                else:
                    self.omegagrid = np.zeros(len(self.data_files))

        elif self.code == "fargo3d":
            self.data_files = [
                fn
                for fn in glob.glob1(self.config["datadir"], "gasdens*.dat")
                if re.match(r"gasdens\d+.dat", fn)
            ]
            nfound = len(glob.glob1(self.config["datadir"], "*.cfg"))
            if nfound == 0:
                raise FileNotFoundError(
                    "*.cfg file (FARGO3D planet parameters) does not exist in '%s' directory"
                    % self.config["datadir"]
                )
            elif nfound > 1:
                raise RuntimeError("found more than one possible .cfg file.")

            cfgfile = glob.glob1(self.config["datadir"], "*.cfg")[0]

            self.cfgconfig = inifix.load(os.path.join(self.config["datadir"], cfgfile))
            # self.h0 = self.iniconfig["ASPECTRATIO"]
            if self.config["isPlanet"]:
                if Path(self.config["datadir"]).joinpath("planet0.dat").is_file():
                    columns = np.loadtxt(
                        os.path.join(self.config["datadir"], "planet0.dat")
                    ).T
                    self.qpl = columns[7]
                    self.dpl = np.sqrt(np.sum(columns[1:4] ** 2, axis=0))
                    self.xpl = columns[1]
                    self.ypl = columns[2]
                    self.tpl = columns[8]
                else:
                    self.qpl = np.full(
                        len(self.data_files), self.cfgconfig[list(self.cfgconfig)[0]][1]
                    )
                    self.dpl = np.full(
                        len(self.data_files), self.cfgconfig[list(self.cfgconfig)[0]][0]
                    )
                self.omegaplanet = np.sqrt(
                    (1.0 + self.qpl) / self.dpl / self.dpl / self.dpl
                )
            if self.config["corotate"]:
                self.vtk = self.iniconfig["NINTERM"] * self.iniconfig["DT"]
                if self.config["isPlanet"]:
                    self.omegagrid = self.omegaplanet
                else:
                    self.omegagrid = np.zeros(len(self.data_files))

        if not self.data_files:
            raise FileNotFoundError("No data files were found.")


class Mesh(InitParamNonos):
    """
    Adapted from Pablo Benitez-Llambay
    Mesh class, for keeping all the mesh data.
    Input: directory [string] -> this is where the domain files are.
    """

    def __init__(self, nonos_config, sim_paramfile=None, **kwargs):
        super().__init__(
            nonos_config=nonos_config, sim_paramfile=sim_paramfile, **kwargs
        )  # All the InitParamNonos attributes inside Field
        super().load()
        logging.debug("mesh parameters: started")
        if self.code == "idefix" or self.code == "pluto":
            first_vtk = next(glob.iglob(os.path.join(self.config["datadir"], "*.vtk")))
            try:
                domain = readVTK(
                    first_vtk,
                    cell="edges",
                    computedata=False,
                )
            except RuntimeError:
                domain = readVTK(
                    first_vtk,
                    geometry="polar",
                    cell="edges",
                    computedata=False,
                )

            self.domain = domain

            self._native_geometry = self.domain.geometry
            if self._native_geometry not in ("cylindrical", "polar"):
                raise NotImplementedError(
                    f"geometry flag '{self._native_geometry}' not implemented yet for readVTK"
                )

            self.nx = self.domain.n1
            self.ny = self.domain.n2
            self.nz = self.domain.n3

            self.xedge = self.domain.x  # X-Edge
            self.yedge = self.domain.y - np.pi  # Y-Edge
            self.zedge = self.domain.z  # Z-Edge

            # index of the cell in the midplane
            self.imidplane = self.nz // 2

        elif self.code == "fargo3d":
            self._native_geometry = self.iniconfig["COORDINATES"]
            nfound_x = len(glob.glob1(self.config["datadir"], "domain_x.dat"))
            if nfound_x != 1:
                raise FileNotFoundError("domain_x.dat not found.")
            nfound_y = len(glob.glob1(self.config["datadir"], "domain_y.dat"))
            if nfound_y != 1:
                raise FileNotFoundError("domain_y.dat not found.")
            nfound_z = len(glob.glob1(self.config["datadir"], "domain_z.dat"))
            if nfound_z != 1:
                raise FileNotFoundError("domain_z.dat not found.")

            domain_x = np.loadtxt(os.path.join(self.config["datadir"], "domain_x.dat"))
            # We avoid ghost cells
            domain_y = np.loadtxt(os.path.join(self.config["datadir"], "domain_y.dat"))[
                3:-3
            ]
            domain_z = np.loadtxt(os.path.join(self.config["datadir"], "domain_z.dat"))
            if domain_z.shape[0] > 6:
                domain_z = domain_z[3:-3]

            self.xedge = domain_y  # X-Edge
            self.yedge = domain_x  # Y-Edge
            # self.zedge = np.pi/2-domain_z #Z-Edge #latitute
            self.zedge = domain_z  # Z-Edge #latitute

            self.nx = len(self.xedge) - 1
            self.ny = len(self.yedge) - 1
            self.nz = len(self.zedge) - 1

            if np.sign(self.zedge[0]) != np.sign(self.zedge[-1]):
                self.imidplane = self.nz // 2
            else:
                self.imidplane = -1

        if self._native_geometry == "cylindrical":
            self._native_geometry = "polar"

        self.xmed = 0.5 * (self.xedge[1:] + self.xedge[:-1])  # X-Center
        self.ymed = 0.5 * (self.yedge[1:] + self.yedge[:-1])  # Y-Center
        self.zmed = 0.5 * (self.zedge[1:] + self.zedge[:-1])  # Z-Center

        # width of each cell in all directions
        self.dx = np.ediff1d(self.xedge)
        self.dy = np.ediff1d(self.yedge)
        self.dz = np.ediff1d(self.zedge)

        self.x = self.xedge
        self.y = self.yedge
        self.z = self.zedge

        # TODO: TEST that when _native_geometry no cylindrical
        if self._native_geometry == "cartesian":
            self._default_point = [0, 0, 0]
        elif self._native_geometry == "spherical":
            self._default_point = [1, np.pi / 2, 0]
        elif self._native_geometry in ("cylindrical", "polar"):
            self._default_point = [1, 0, 0]
        else:
            raise RuntimeError(f"Got unknown geometry flag '{self._native_geometry}'")

        logging.debug("mesh parameters: finished")

    @property
    def coord(self):
        return self.x, self.y, self.z

    @property
    def coordmed(self):
        return self.xmed, self.ymed, self.zmed


class FieldNonos(Mesh, InitParamNonos):
    """
    Inspired by Pablo Benitez-Llambay
    Field class, it stores the mesh, parameters and scalar data
    for a scalar field.
    Input: field [string] -> filename of the field
           directory='' [string] -> where filename is
    """

    def __init__(self, init, sim_paramfile=None, check=True, **kwargs):
        self.check = check
        self.init = init

        Mesh.__init__(
            self, nonos_config=self.init.config, sim_paramfile=sim_paramfile, **kwargs
        )  # All the Mesh attributes inside Field
        InitParamNonos.__init__(
            self, nonos_config=self.init.config, sim_paramfile=sim_paramfile, **kwargs
        )  # All the InitParamNonos attributes inside Field

        if isinstance(self.config["on"], Sequence):
            self.on = self.config["on"][0]
        else:
            self.on = self.config["on"]

        field = self.config["field"]
        if self.code != "idefix":
            field = field.lower()

        if self.code == "fargo3d":
            known_aliases = {"rho": "dens", "vx1": "vy", "vx2": "vx", "vx3": "vz"}
            field = known_aliases[field]
            filedata = "gas%s%d.dat" % (field, self.on)
            filedata0 = "gas%s0.dat" % field
        else:
            # Idefix or Pluto
            filedata = "data.%04d.vtk" % self.on
            filedata0 = "data.0000.vtk"

        self.field = field

        # FIXME: reactivate this warning
        # if self.config["corotate"] and not(self.config["isPlanet"]):
        #    warnings.warn("We don't rotate the grid if there is no planet for now.\nomegagrid = 0.")

        datafile = os.path.join(self.config["datadir"], filedata)

        if not os.path.isfile(datafile):
            raise FileNotFoundError(datafile)
        self.data = self.__open_field(datafile)  # The scalar data is here.

        if self.config["diff"]:
            datafile = os.path.join(self.config["datadir"], filedata0)
            if not os.path.isfile(datafile):
                raise FileNotFoundError(datafile)
            self.data0 = self.__open_field(datafile)

        if self.config["log"]:
            if self.config["diff"]:
                self.data = np.log10(self.data / self.data0)
                self.title = fr"log($\frac{{{self.field}}}{{{self.field}_0}}$)"
            else:
                self.data = np.log10(self.data)
                self.title = "log(%s)" % self.field
        else:
            if self.config["diff"]:
                self.data = (self.data - self.data0) / self.data0
                self.title = r"$\frac{{{} - {}_0}}{{{}_0}}$".format(
                    self.field,
                    self.field,
                    self.field,
                )
            else:
                self.data = self.data
                self.title = "%s" % self.field

    def __open_field(self, f):
        """
        Reading the data
        """
        super().load()
        if self.code == "fargo3d":
            if self._native_geometry in ("cylindrical", "polar"):
                data = np.fromfile(f, dtype="float64")
                data = (data.reshape(self.nz, self.nx, self.ny)).transpose(
                    1, 2, 0
                )  # rad, pĥi, theta
            else:
                raise NotImplementedError(
                    f"geometry flag '{self._native_geometry}' not implemented yet for readVTK"
                )
        else:
            # Idefix or Pluto
            if self._native_geometry in ("cylindrical", "polar"):
                data = (
                    readVTK(f, geometry=self._native_geometry, cell="edges")
                    .data[self.field]
                    .astype(np.float32)
                )
                data = np.concatenate(
                    (data[:, self.ny // 2 : self.ny, :], data[:, 0 : self.ny // 2, :]),
                    axis=1,
                )
            else:
                raise NotImplementedError(
                    f"geometry flag '{self._native_geometry}' not implemented yet for readVTK"
                )

        """
        if we try to rotate a grid at 0 speed
        and if the domain is exactly [-pi,pi],
        impossible to perform the following calculation (try/except)
        we therefore don't move the grid if the rotation speed is null
        """
        if not (
            self.config["corotate"]
            and abs(self.vtk * sum(self.omegagrid[: self.on])) > 1.0e-16
        ):
            return data

        logging.debug("grid rotation: started")
        P, R = np.meshgrid(self.y, self.x)
        ind_on = find_nearest(self.tpl, self.vtk * self.on)
        Prot = P - (np.arctan2(self.ypl, self.xpl)[ind_on]) % (2 * np.pi)
        # Prot = P - (self.vtk * sum(self.omegagrid[: self.on])) % (2 * np.pi)
        try:
            index = (np.where(Prot[0] > np.pi))[0].min()
        except ValueError:
            index = (np.where(Prot[0] < -np.pi))[0].max()
        if self._native_geometry in ("cylindrical", "polar"):
            data = np.concatenate(
                (data[:, index : self.ny, :], data[:, 0:index, :]), axis=1
            )
        else:
            raise NotImplementedError(
                f"geometry flag '{self._native_geometry}' not implemented yet for readVTK"
            )
        logging.debug("grid rotation: finished")
        return data


class PlotNonos(FieldNonos):
    """
    Plot class which uses Field to compute different graphs.
    """

    def axiplot(self, ax, *, vmin=None, vmax=None, average=None, extent=None, **karg):
        if average is None:
            average = self.init.config["average"]
        # if average:
        #     dataRZ = np.mean(self.data, axis=1)
        #     dataR = np.mean(dataRZ, axis=1) * next(
        #         item for item in [self.z.max() - self.z.min(), 1.0] if item != 0
        #     )
        #     dataProfile = dataR
        # else:
        #     dataRZ = self.data[:, self.ny // 2, :]
        #     dataR = dataRZ[:, self.imidplane]
        #     dataProfile = dataR

        # TODO: TEST + change for different data _native_geometry
        if self._native_geometry in ("cylindrical", "polar"):
            if average:
                if self.data.shape[1] <= 1:
                    dataRZ = self.data[:, 0, :]
                else:
                    dataRZ = (
                        np.sum(
                            (
                                ((self.data[:, 1:, :] + self.data[:, :-1, :]) / 2)
                                * np.ediff1d(self.coordmed[1])[None, :, None]
                            ),
                            axis=1,
                        )
                        / 2.0
                        / np.pi
                    )
                dataR = np.sum(
                    (
                        ((dataRZ[:, 1:] + dataRZ[:, :-1]) / 2)
                        * np.ediff1d(self.coordmed[2])[None, :]
                    ),
                    axis=1,
                )

                # dataRZ = choosemean(self.data, (1,3,2), self.coord)
                # dataR = choosemean(dataR, (1,3,2), self.coord)
                dataProfile = dataR
            else:
                dataRZ = self.data[:, self.ny // 2, :]
                dataR = dataRZ[:, self.imidplane]
                dataProfile = dataR
        else:
            raise NotImplementedError(
                f"geometry flag '{self._native_geometry}' not implemented yet for axiplot"
            )

        vmin, vmax = parse_vmin_vmax(
            vmin, vmax, diff=self.config["diff"], data=dataProfile
        )

        ax.plot(self.xmed, dataProfile, **karg)

        extent = parse_range(extent, dim=1)
        extent = range_converter(extent, abscissa=self.xmed, ordinate=np.zeros(2))

        logging.debug("xmin: %f" % extent[0])
        logging.debug("xmax: %f" % extent[1])

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(vmin, vmax)
        ax.set_xlabel("Radius")
        ax.set_ylabel(self.title)

        self.xplot = self.xmed
        self.yplot = np.empty(0)
        self.dataplot = dataProfile

    def plot(
        self,
        ax,
        *,
        vmin=None,
        vmax=None,
        plane=(1, 2, 3),  # default: (x,y)
        geometry="cartesian",
        func_proj=None,
        average=None,
        lic=None,
        licres=None,
        scaling=1,
        cmap=None,
        extent=None,
        **karg,
    ):
        """
        A layer for pcolormesh function.
        """
        vmin, vmax = parse_vmin_vmax(
            vmin, vmax, diff=self.config["diff"], data=self.data
        )

        extent = parse_range(extent, dim=2)

        if average is None:
            average = self.init.config["average"]
        if cmap is None:
            cmap = self.init.config["cmap"]

        # TODO: below: works for a cylindrical _native_geometry -> TEST + change for other _native_geometries
        if self._native_geometry in ("cylindrical", "polar"):
            if geometry == "cylindrical":
                ax.set_aspect("auto")
                if plane[:-1] == (1, 2):
                    ax.set_ylabel(r"$\phi$ [c.u.]")
                    ax.set_xlabel("R [c.u.]")
                elif plane[:-1] == (1, 3):
                    ax.set_ylabel("z [c.u.]")
                    ax.set_xlabel("R [c.u.]")
                else:
                    raise NotImplementedError(
                        f"plane {plane[:-1]} is not implemented yet in a {geometry} projection."
                    )
            elif geometry == "cartesian":
                if plane[:-1] == (1, 2):
                    ax.set_aspect("equal")
                    ax.set_ylabel("Y [c.u.]")
                    ax.set_xlabel("X [c.u.]")
                elif plane[:-1] == (1, 3):
                    ax.set_aspect("auto")
                    ax.set_ylabel("Z [c.u.]")
                    ax.set_xlabel("X [c.u.]")
                elif plane[:-1] == (2, 3):
                    ax.set_aspect("auto")
                    ax.set_ylabel("Z [c.u.]")
                    ax.set_xlabel("Y [c.u.]")
                else:
                    raise NotImplementedError(
                        f"plane {plane[:-1]} is not implemented yet in a {geometry} projection."
                    )
            elif geometry == "spherical":
                ax.set_aspect("auto")  # for now
                ax.set_ylabel(r"$\theta$ [c.u.]")
                ax.set_xlabel("r [c.u.]")
            else:
                raise ValueError(f"Unknown geometry '{geometry}'")

            if (1 in plane[:-1]) and (self.x.shape[0] <= 1):
                raise IndexError("No radial direction, the simulation is not 3D.")
            if (2 in plane[:-1]) and (self.y.shape[0] <= 1):
                raise IndexError("No azimuthal direction, the simulation is not 3D.")
            if (3 in plane[:-1]) and (self.z.shape[0] <= 1):
                raise IndexError("No vertical direction, the simulation is not 3D.")
        else:
            raise NotImplementedError(
                f"geometry flag '{self._native_geometry}' not implemented yet for plot"
            )

        logging.debug("pcolormesh: started")

        # If we plot a slice, we then need to adapt the data itself
        # to be coherent with the projection plan.
        # We look for the index of the 3d dimension 1D array
        # that matches self._default_point
        if average:
            data = choosemean(self.data, plane, self.coord)
        else:
            data = chooseslice(self.data, plane, self.coord, self._default_point)

        # convert 1D coordinates arrays (self.coord)
        # into 2D coordinates arrays (coordgrid) via meshgrid,
        # using (plane[0],plane[1]) for the projection plane
        # and self._default_point for the 3d (plane[2]) dimension
        coordgrid = meshgrid_from_plane(
            self.coord, plane[0], plane[1], self._default_point
        )
        # reorder the coordinates in order to prepare the transformation from a native geometry to a new geometry
        # coordgrid = tuple(np.array(coordgrid, dtype=object)[np.array(plane) - 1])
        coordgrid = tuple(coordgrid[_] for _ in np.argsort(plane))

        # (coord[0]=R,coord[1]=phi by default)
        # then, depending on the geometry,
        # we transform these 2D coordinates arrays
        if func_proj is not None:
            coordgrid = func_proj(*coordgrid)

        # TODO: careful, works for a cylindrical _native_geometry
        # but may be wrong when other _native_geometries
        # are implemented (in particular the extent of interpolation)
        if self._native_geometry in ("cylindrical", "polar"):
            if is_set(lic) and self.code in ("pluto", "idefix"):
                if 1 in plane[:-1]:
                    if 3 in plane[:-1]:
                        extent_i = (None, extent[1], extent[2], extent[3])
                    else:
                        extent_i = (None, extent[1], None, None)
                elif 3 in plane[:-1]:
                    extent_i = (None, None, extent[2], extent[3])
                else:
                    extent_i = (None, None, None, None)

                xi, yi, lici = LICstream(
                    self.init,
                    self.on,
                    plane=plane,
                    lines=lic,
                    func_proj=func_proj,
                    corotate=self.config["corotate"],
                    isPlanet=self.config["isPlanet"],
                    average=average,
                    xxmin=extent_i[0],
                    xxmax=extent_i[1],
                    yymin=extent_i[2],
                    yymax=extent_i[3],
                    dxx=licres,
                    dyy=licres,
                    kernel_length=30,
                    niter=2,
                )
                # print(f"xmin: {xi.min()}, xmax: {xi.max()}, ymin: {yi.min()}, ymax: {yi.max()}")
                # print(f"extent: {extent}")
                coordgridmed = meshgrid_from_plane(
                    self.coordmed, plane[0], plane[1], self._default_point
                )
                datai = interpol(
                    coordgridmed[0],
                    coordgridmed[1],
                    data,
                    method="nearest",
                    dxx=licres,
                    dyy=licres,
                    xxmin=extent_i[0],
                    xxmax=extent_i[1],
                    yymin=extent_i[2],
                    yymax=extent_i[3],
                )[2]
                if self.config["log"]:
                    datalic = np.log10(lici) + datai
                else:
                    datalic = lici * datai
                im = ax.pcolormesh(
                    xi,
                    yi,
                    datalic,
                    cmap=cb.cbmap(palette=cmap),
                    vmin=vmin,
                    vmax=vmax,
                    **karg,
                )

                extent = range_converter(
                    extent,
                    abscissa=xi,
                    ordinate=yi,
                )

            else:
                im = ax.pcolormesh(
                    coordgrid[plane[0] - 1],
                    coordgrid[plane[1] - 1],
                    data,
                    cmap=cb.cbmap(palette=cmap),
                    vmin=vmin,
                    vmax=vmax,
                    **karg,
                )

                extent = range_converter(
                    extent,
                    abscissa=coordgrid[plane[0] - 1],
                    ordinate=coordgrid[plane[1] - 1],
                )

            if self.init.config["grid"]:
                ax.plot(
                    coordgrid[plane[0] - 1], coordgrid[plane[1] - 1], c="k", linewidth=1
                )
                ax.plot(
                    coordgrid[plane[0] - 1].transpose(),
                    coordgrid[plane[1] - 1].transpose(),
                    c="k",
                    linewidth=1,
                )

            logging.debug("xmin: %f" % extent[0])
            logging.debug("xmax: %f" % extent[1])
            logging.debug("ymin: %f" % extent[2])
            logging.debug("ymax: %f" % extent[3])

            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

            ax.set_title(self.code)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax, orientation="vertical")
            cbar.set_label(self.title)

        else:
            raise NotImplementedError(
                f"geometry flag '{self._native_geometry}' not implemented yet for plot"
            )

        logging.debug("pcolormesh: finished")

        if is_set(lic) and self.code in ("pluto", "idefix"):
            self.xplot = xi
            self.yplot = yi
            self.dataplot = datalic
        else:
            self.xplot = coordgrid[plane[0] - 1]
            self.yplot = coordgrid[plane[1] - 1]
            self.dataplot = data


def is_averageSafe(sigma0, sigmaSlope, plot=False):
    init = InitParamNonos()  # initialize the major parameters
    fieldon = FieldNonos(
        init, field="RHO", on=0
    )  # fieldon object with the density field at on=0
    datarz = np.mean(fieldon.data, axis=1)  # azimuthally-averaged density field
    error = (
        sigma0 * pow(fieldon.xmed, -sigmaSlope)
        - np.mean(datarz, axis=1)
        * next(item for item in [fieldon.z.max() - fieldon.z.min(), 1.0] if item != 0)
    ) / (
        sigma0 * pow(fieldon.xmed, -sigmaSlope)
    )  # comparison between Sigma(R) profile and integral of rho(R,z) between zmin and zmax
    if any(100 * abs(error) > 3):
        print_warn(
            "With a maximum of %.1f percents of error, the averaging procedure may not be safe.\nzmax/h is probably too small.\nUse rather average=False (-slice) or increase zmin/zmax."
            % np.max(100 * abs(error))
        )
    else:
        print_warn(
            "Only %.1f percents of error maximum in the averaging procedure."
            % np.max(100 * abs(error))
        )
    if plot:
        fig, ax = plt.subplots()
        ax.plot(
            fieldon.xmed,
            np.mean(datarz, axis=1)
            * next(
                item for item in [fieldon.z.max() - fieldon.z.min(), 1.0] if item != 0
            ),
            label=r"$\int_{z_{min}}^{z_{max}} \rho(R,z)dz$ = (z$_{max}$-z$_{min}$)$\langle\rho\rangle_z$",
        )
        ax.plot(
            fieldon.xmed,
            sigma0 * pow(fieldon.xmed, -sigmaSlope),
            label=r"$\Sigma_0$R$^{-\sigma}$",
        )
        # ax.plot(fieldon.xmed, np.mean(datarz, axis=1)*(fieldon.z.max()-fieldon.z.min()), label='integral of data using mean and zmin/zmax')
        # ax.plot(fieldon.xmed, sigma0*pow(fieldon.xmed,-sigmaSlope), label='theoretical reference')

        ax.set_ylabel(r"$\Sigma_0(R)$")
        ax.set_xlabel("Radius")
        ax.tick_params("both")
        ax.legend(frameon=False, prop={"size": 10, "family": "monospace"})
        fig2, ax2 = plt.subplots()
        ax2.plot(fieldon.xmed, abs(error) * 100)
        ax2.set_ylabel(r"Error (%)")
        ax2.set_xlabel("Radius")
        plt.show()


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def interpol(
    xx,
    yy,
    u,
    *,
    method="linear",
    xxmin=None,
    xxmax=None,
    yymin=None,
    yymax=None,
    dxx=None,
    dyy=None,
):
    if xxmin is None:
        xxmin = xx.min()
    if xxmax is None:
        xxmax = xx.max()
    if yymin is None:
        yymin = yy.min()
    if yymax is None:
        yymax = yy.max()

    x = np.linspace(xxmin, xxmax, dxx * xx.shape[0])
    y = np.linspace(yymin, yymax, dyy * yy.shape[1])

    xi, yi = np.meshgrid(x, y)

    # then, interpolate your data onto this grid:

    px = xx.flatten()
    py = yy.flatten()
    pu = u.flatten()

    gu = griddata((px, py), pu, (xi, yi), method=method)

    return (x, y, gu)


def chooseslice(field, plane, coord, _default_point):
    if plane[2] - 1 == 0:
        fieldslice = field[
            find_nearest(coord[plane[2] - 1], _default_point[plane[2] - 1]), :, :
        ]
    elif plane[2] - 1 == 1:
        fieldslice = field[
            :, find_nearest(coord[plane[2] - 1], _default_point[plane[2] - 1]), :
        ]
    elif plane[2] - 1 == 2:
        fieldslice = field[
            :, :, find_nearest(coord[plane[2] - 1], _default_point[plane[2] - 1])
        ]
    else:
        raise ValueError("Plane not defined. Should be any permutation of (1,2,3).")

    return fieldslice


def coordiff3d(coord, plane):
    coordiff = np.ediff1d(coord[plane[2] - 1])
    if plane[2] - 1 == 0:
        coordiff3d = coordiff[:, None, None]
    elif plane[2] - 1 == 1:
        coordiff3d = coordiff[None, :, None]
    elif plane[2] - 1 == 2:
        coordiff3d = coordiff[None, None, :]
    else:
        raise ValueError("Plane not defined. Should be any permutation of (1,2,3).")
    return coordiff3d


def choosemean(field, plane, coord):
    span = coord[plane[2] - 1].ptp() or 1.0
    # fieldmean = np.mean(field, axis=plane[2] - 1) * span
    if plane[2] == 2:
        fieldmean = np.sum(field * coordiff3d(coord, plane), axis=plane[2] - 1) / span
    else:
        fieldmean = np.sum(field * coordiff3d(coord, plane), axis=plane[2] - 1)
    return fieldmean


def LICstream(
    init,
    on,
    *,
    plane=(1, 2, 3),
    lines="V",
    func_proj=None,
    corotate=False,
    isPlanet=False,
    average=False,
    datadir="",
    xxmin=None,
    xxmax=None,
    yymin=None,
    yymax=None,
    dxx=None,
    dyy=None,
    kernel_length=30,
    niter=2,
):
    lx1on, lx2on = (
        FieldNonos(
            init,
            field=f"{lines}X{i}",
            on=on,
            diff=False,
            log=False,
            corotate=corotate,
            isPlanet=isPlanet,
            datadir=datadir,
            check=False,
        )
        for i in plane[:-1]
    )

    lx1 = lx1on.data.astype(np.float32)
    # TODO: change/generalize this,
    # as it works for a cylindrical _native_geometry,
    # but not a spherical one
    if lx2on._native_geometry in ("cylindrical", "polar"):
        if isPlanet and (2 in plane[:-1]) and (lines == "V"):
            lx2 = (
                lx2on.data - lx2on.omegaplanet[on] * lx2on.xmed[:, None, None]
            ).astype(np.float32)
        else:
            lx2 = lx2on.data.astype(np.float32)
    else:
        raise NotImplementedError(
            f"geometry flag '{lx2on._native_geometry}' not implemented yet for LICstream"
        )

    coordgridmed = meshgrid_from_plane(
        lx1on.coordmed, plane[0], plane[1], lx1on._default_point
    )

    if average:
        lx1avr = choosemean(lx1, plane, lx1on.coord)
        lx2avr = choosemean(lx2, plane, lx1on.coord)
        lx = [lx1avr, lx2avr]
    else:
        lx1slice = chooseslice(lx1, plane, lx1on.coord, lx1on._default_point)
        lx2slice = chooseslice(lx2, plane, lx1on.coord, lx1on._default_point)
        lx = [lx1slice, lx2slice]

    xi, yi, lx1i = interpol(
        coordgridmed[0],
        coordgridmed[1],
        lx[0],
        dxx=dxx,
        dyy=dyy,
        xxmin=xxmin,
        xxmax=xxmax,
        yymin=yymin,
        yymax=yymax,
    )
    lx2i = interpol(
        coordgridmed[0],
        coordgridmed[1],
        lx[1],
        dxx=dxx,
        dyy=dyy,
        xxmin=xxmin,
        xxmax=xxmax,
        yymin=yymin,
        yymax=yymax,
    )[2]

    lx1i = lx1i.astype(np.float32)
    lx2i = lx2i.astype(np.float32)

    texture = random_noise(
        np.zeros((lx1i.shape[0], lx1i.shape[1])),
        mode="gaussian",
        mean=0.5,
        var=0.001,
        seed=0,
    ).astype(np.float32)
    kernel = np.ones(kernel_length).astype(np.float32)

    image = lic_internal.line_integral_convolution(lx1i, lx2i, texture, kernel)
    image_eq = exposure.equalize_hist(image)

    image_relic_eq = image_eq
    for _ in range(niter - 1):
        image_relic_eq = lic_internal.line_integral_convolution(
            lx1i, lx2i, image_relic_eq.astype(np.float32), kernel
        )
    image_relic_eq /= image_relic_eq.max()

    xiedge = 0.5 * (xi[1:] + xi[:-1])
    yiedge = 0.5 * (yi[1:] + yi[:-1])
    xiedge = np.concatenate(
        ([lx1on.coord[plane[0] - 1][0]], xiedge, [lx1on.coord[plane[0] - 1][-1]])
    )
    yiedge = np.concatenate(
        ([lx1on.coord[plane[1] - 1][0]], yiedge, [lx1on.coord[plane[1] - 1][-1]])
    )

    Xi, Yi = np.meshgrid(xiedge, yiedge)
    Zi = lx1on._default_point[plane[2] - 1]
    coordi = (Xi, Yi, Zi)
    # from plane to R,phi,z

    # reorder the coordinates in order to prepare the transformation from a native geometry to a new geometry
    # coordgridedge = tuple(np.array([Xi, Yi, Zi], dtype=object)[np.array(plane) - 1])
    coordgridedge = tuple(coordi[_] for _ in np.argsort(plane))

    if func_proj is not None:
        coordgridedge = func_proj(*coordgridedge)

    return (coordgridedge[plane[0] - 1], coordgridedge[plane[1] - 1], image_relic_eq)


# process function for parallisation purpose with progress bar
# counterParallel = Value('i', 0) # initialization of a counter
def process_field(
    on,
    init,
    dim,
    field,
    plane,
    geometry,
    func_proj,
    avr,
    diff,
    log,
    corotate,
    lic,
    licres,
    extent,
    vmin,
    vmax,
    scaling: float,
    cmap,
    isPlanet,
    pbar,
    parallel,
    datadir,
    show: bool,
    dpi: int,
    fmt: str,
    binary: bool,
):
    set_mpl_style(scaling=scaling)

    ploton = PlotNonos(
        init,
        field=field,
        on=on,
        diff=diff,
        log=log,
        corotate=corotate,
        isPlanet=isPlanet,
        datadir=datadir,
        check=False,
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=False)

    # plot the field
    if dim == 2:
        ploton.plot(
            ax,
            vmin=vmin,
            vmax=vmax,
            plane=plane,
            geometry=geometry,
            func_proj=func_proj,
            lic=lic,
            licres=licres,
            average=avr,
            cmap=cmap,
            extent=extent,
        )
        # TODO: TEST that when _native_geometry different from cylindrical
        prefix = get_keys_from_geomtransforms(
            GEOM_TRANSFORMS[ploton._native_geometry], [plane, geometry]
        )

    # plot the 1D profile
    elif dim == 1:
        ploton.axiplot(ax, vmin=vmin, vmax=vmax, average=avr, extent=extent)
        prefix = "axi"

    if show:
        plt.show()
    elif binary:
        logging.debug("saving binary file: started")
        fnamenpz = f"{prefix}{'_avr' if avr else '_slice'}_{field}{f'_lic{lic}_' if is_set(lic) else ''}{'_diff' if diff else ''}{'_log' if log else ''}{geometry if dim==2 else ''}{on:04d}"
        np.savez_compressed(
            fnamenpz, abs=ploton.xplot, ord=ploton.yplot, field=ploton.dataplot
        )
        logging.debug("saving binary file: finished")
    else:
        logging.debug("saving plot: started")
        filename = f"{prefix}{'_avr' if avr else '_slice'}_{field}{f'_lic{lic}_' if is_set(lic) else ''}{'_diff' if diff else ''}{'_log' if log else ''}{geometry if dim==2 else ''}{on:04d}.{fmt}"
        fig.savefig(filename, bbox_inches="tight", dpi=dpi)
        logging.debug("saving plot: finished")
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nonos",
        description=__doc__,
    )

    parser.add_argument(
        "-dir",
        dest="datadir",
        help=f"location of output files and param files (default: '{DEFAULTS['datadir']}').",
    )
    parser.add_argument(
        "-field",
        choices=["RHO", "VX1", "VX2", "VX3", "BX1", "BX2", "BX3", "PRS"],
        help=f"name of field to plot (default: '{DEFAULTS['field']}').",
    )
    parser.add_argument(
        "-plane",
        choices=["rphi", "rz", "rtheta", "xy", "xz", "yz"],
        help=f"name of plane of projection (default: '{DEFAULTS['plane']}').",
    )
    parser.add_argument(
        "-range",
        type=str,
        nargs="+",
        help=f"range of matplotlib window (default: {DEFAULTS['range']}), example: x x -2 2",
    )
    parser.add_argument(
        "-vmin",
        type=float,
        help=f"min value in -diff mode (default: {DEFAULTS['vmin']})",
    )
    parser.add_argument(
        "-vmax",
        type=float,
        help=f"max value in -diff mode (default: {DEFAULTS['vmax']})",
    )
    parser.add_argument(
        "-cpu",
        "-ncpu",
        dest="ncpu",
        type=int,
        help=f"number of parallel processes (default: {DEFAULTS['ncpu']}).",
    )

    select_group = parser.add_mutually_exclusive_group()
    select_group.add_argument(
        "-on",
        type=int,
        nargs="+",
        help="output number(s) (on) to plot. "
        "This can be a single value or a range (start, end, [step]) where both ends are inclusive. "
        "(default: last output available).",
    )
    select_group.add_argument(
        "-all",
        action="store_true",
        help="save an image for every available snapshot (this will force show=False).",
    )

    # boolean flags use False as a default value (...by default)
    # forcing them to None instead will allow them to be pruned
    # when we build the ChainMap config
    flag_group = parser.add_argument_group("boolean flags")
    flag_group.add_argument(
        "-diff",
        action="store_true",
        default=None,
        help="plot the relative perturbation of the field f, i.e. (f-f0)/f0.",
    )
    flag_group.add_argument(
        "-log",
        action="store_true",
        default=None,
        help="plot the log10 of the field f, i.e. log(f).",
    )
    flag_group.add_argument(
        "-isp",
        dest="isPlanet",
        action="store_true",
        default=None,
        help="is there a planet in the grid ?",
    )
    flag_group.add_argument(
        "-corotate",
        action="store_true",
        default=None,
        help="does the grid corotate? Works in pair with -isp.",
    )
    flag_group.add_argument(
        "-grid", action="store_true", default=None, help="show the computational grid."
    )
    flag_group.add_argument(
        "-slice",
        action="store_true",
        default=None,
        help="perform a slice along the third dimension. ",
    )
    flag_group.add_argument(
        "-pbar",
        dest="progressBar",
        action="store_true",
        default=None,
        help="display a progress bar",
    )

    stream_group = parser.add_argument_group("streamlines options")
    stream_group.add_argument(
        "-lic",
        choices=["V", "B"],
        help=f"which vector field for lic streamlines (default: '{DEFAULTS['lic']}')",
    )
    stream_group.add_argument(
        "-licres",
        type=int,
        help=f"lic interpolation cell refinement (default: {DEFAULTS['licres']})",
    )

    parser.add_argument(
        "-dim",
        dest="dimensionality",
        type=int,
        choices=[1, 2],
        help="dimensionality in projection: 1 for a line plot, 2 (default) for a map.",
    )
    parser.add_argument(
        "-scaling",
        dest="scaling",
        type=float,
        help=f"scale the overall sizes of features in the graph (fonts, linewidth...) (default: {DEFAULTS['scaling']}).",
    )
    parser.add_argument(
        "-cmap",
        help=f"choice of colormap for the -dim 2 maps (default: '{DEFAULTS['cmap']}').",
    )
    parser.add_argument(
        "-fmt",
        "-format",
        dest="format",
        help=f"select output image file format (default: {DEFAULTS['format']})",
    )
    parser.add_argument(
        "-dpi",
        type=int,
        help="image file resolution (default: DEFAULTS['dpi'])",
    )

    cli_only_group = parser.add_argument_group("CLI-only options")
    cli_input_group = cli_only_group.add_mutually_exclusive_group()
    cli_input_group.add_argument(
        "-input", "-i", dest="input", type=str, help="specify a configuration file."
    )
    cli_input_group.add_argument(
        "-isolated", action="store_true", help="ignore any existing 'nonos.toml' file."
    )
    cli_action_group = cli_only_group.add_mutually_exclusive_group()
    cli_action_group.add_argument(
        "-d",
        "-display",
        dest="display",
        action="store_true",
        help="open a graphic window with the plot (only works with a single image)",
    )
    cli_action_group.add_argument(
        "-bin",
        dest="binary",
        action="store_true",
        default=None,
        help="create a binary file",
    )
    cli_action_group.add_argument(
        "-version",
        "--version",
        action="store_true",
        help="show raw version number and exit",
    )
    cli_action_group.add_argument(
        "-logo",
        action="store_true",
        help="show Nonos logo with version number, and exit.",
    )
    cli_action_group.add_argument(
        "-config", action="store_true", help="show configuration and exit."
    )

    cli_debug_group = cli_only_group.add_mutually_exclusive_group()
    cli_debug_group.add_argument(
        "-v",
        "-verbose",
        "--verbose",
        action="count",
        default=0,
        help="increase output verbosity (-v: info, -vv: debug).",
    )

    clargs = vars(parser.parse_args(argv))

    # special cases: destructively consume CLI-only arguments with dict.pop

    if clargs.pop("logo"):
        with open(pkg_resources.resource_filename("nonos", "logo.txt")) as fh:
            logo = fh.read()
        print(f"{logo}{__doc__}Version {__version__}")
        return 0

    if clargs.pop("version"):
        print(__version__)
        return 0

    level = parse_verbose_level(clargs.pop("verbose"))

    FORMAT = "%(message)s"
    logging.basicConfig(
        level=level,
        force=True,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler()],
    )

    if clargs.pop("isolated"):
        config_file_args: Dict[str, Any] = {}
    elif (ifile := clargs.pop("input")) is not None:
        if not os.path.isfile(ifile):
            print_err(f"Couldn't find requested input file '{ifile}'.")
            return 1
        print_warn(f"[bold white]Using parameters from '{ifile}'.")
        config_file_args = toml.load(ifile)
    elif os.path.isfile("nonos.toml"):
        print_warn("[bold white]Using parameters from 'nonos.toml'.")
        config_file_args = toml.load("nonos.toml")
    else:
        config_file_args = {}

    # check that every parameter in the configuration is also exposed to the CLI
    assert not set(DEFAULTS).difference(set(clargs))

    # squeeze out any unset value form cli config to leave room for file parameters
    clargs = {k: v for k, v in clargs.items() if v is not None}

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
        init = InitParamNonos(nonos_config=args)
        mesh = Mesh(nonos_config=args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print_err(exc)
        return 1
    init.load()

    plane, geometry, func_proj = GEOM_TRANSFORMS[mesh._native_geometry][args["plane"]]

    available = set()
    for fn in init.data_files:
        if (num := re.search(r"\d+", fn)) is not None:
            available.add(int(num.group()))

    if args.pop("all"):
        requested = available
    else:
        try:
            requested = set(
                parse_output_number_range(args["on"], maxval=max(available))
            )
        except ValueError as exc:
            print_err(exc)
            return 1

    if not (toplot := list(requested.intersection(available))):
        print_err(
            f"No requested output file was found (requested {requested}, found {available})."
        )
        return 1
    args["on"] = toplot

    if (show := clargs.pop("display")) and len(args["on"]) > 1:
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

    args["field"] = args["field"].upper()

    if args["corotate"] and not args["isPlanet"]:
        print_warn(
            "We don't rotate the grid if there is no planet for now.\nomegagrid = 0."
        )

    if not is_set(args["vmin"]) or not is_set(args["vmax"]):
        ref_on = args["on"][len(args["on"]) // 2]
        fieldon = FieldNonos(init, on=ref_on, check=False)

        if args["dimensionality"] == 2:
            data = fieldon.data
        elif args["dimensionality"] == 1:
            data = np.mean(np.mean(fieldon.data, axis=1), axis=1)

        vmin, vmax = parse_vmin_vmax(args["vmin"], args["vmax"], args["diff"], data)
    else:
        vmin, vmax = args["vmin"], args["vmax"]

    extent = args["range"]

    if args["ncpu"] > (ncpu := min(args["ncpu"], os.cpu_count())):
        print_warn(
            f"Requested {args['ncpu']}, but the runner only has access to {ncpu}."
        )

    if args["progressBar"]:
        from rich.progress import track
    else:
        # replace rich.progress.track with a no-op dummy
        def track(iterable, *args, **kwargs):
            return iterable

    # call of the process_field function, whether it be in parallel or not
    # TODO: reduce this to the bare minimum
    func = functools.partial(
        process_field,
        init=init,
        dim=args["dimensionality"],
        field=args["field"],
        plane=plane,
        geometry=geometry,
        func_proj=func_proj,
        avr=not args["slice"],
        diff=args["diff"],
        log=args["log"],
        corotate=args["corotate"],
        lic=args["lic"],
        licres=args["licres"],
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        scaling=args["scaling"],
        cmap=args["cmap"],
        isPlanet=args["isPlanet"],
        pbar=args["progressBar"],
        parallel=args["ncpu"] > 1,
        datadir=args["datadir"],
        show=show,
        dpi=args["dpi"],
        fmt=args["format"],
        binary=args["binary"],
    )

    tstart = time.time()
    with Pool(ncpu) as pool:
        list(
            track(
                pool.imap(func, args["on"]),
                description="Processing snapshots",
                total=len(args["on"]),
            )
        )
    if not show:
        logging.info("Operation took %.2fs" % (time.time() - tstart))
    # current, peak = tracemalloc.get_traced_memory()
    # print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    # tracemalloc.stop()
    return 0
