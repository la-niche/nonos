import glob
import os
import re
from pathlib import Path
from typing import Union

import gpgi
import inifix
import numpy as np

from nonos.logging import logger

_AnyArray = Union[np.ndarray, np.memmap]

_INIFILES_LOOKUP_TABLE = {
    "idefix.ini": "idefix",
    "pluto.ini": "pluto",
    "variables.par": "fargo3d",
}


class Parameters:
    def __init__(
        self,
        *,
        inifile: str = "",
        code: str = "",
        directory: str = "",
    ):
        self.directory = directory
        self.paramfile = inifile
        self.code = code
        if (self.code, self.paramfile).count("") == 0:
            found = Path(self.directory).joinpath(self.paramfile).is_file()
            if not found:
                raise FileNotFoundError(f"{self.paramfile} not found.")
        elif (self.code, self.paramfile).count("") == 2:
            found_dict = {
                paramfile: Path(self.directory).joinpath(paramfile).is_file()
                for paramfile in _INIFILES_LOOKUP_TABLE
            }
            nfound = sum(list(found_dict.values()))
            if nfound == 0:
                raise FileNotFoundError(
                    f"{', '.join(_INIFILES_LOOKUP_TABLE.keys())} not found."
                )
            elif nfound > 1:
                raise RuntimeError("found more than one possible ini file.")
            self.paramfile = list(_INIFILES_LOOKUP_TABLE.keys())[
                list(found_dict.values()).index(True)
            ]
            self.code = _INIFILES_LOOKUP_TABLE[self.paramfile]
        elif (self.code, self.paramfile).count("") == 1:
            raise ValueError("both inifile and code have to be given.")

    def loadIniFile(self):
        self.inifile = inifix.load(os.path.join(self.directory, self.paramfile))
        if self.code == "idefix":
            self.vtk = self.inifile["Output"]["vtk"]
            try:
                self.omegaframe = self.inifile["Hydro"]["rotation"]
                self.frame = "F"
            except KeyError:
                self.omegaframe = None
                self.frame = None
        elif self.code == "pluto":
            self.vtk = self.inifile["Static Grid Output"]["vtk"][0]
            self.omegaframe = None
            self.frame = None
        elif self.code == "fargo3d":
            self.vtk = self.inifile["NINTERM"] * self.inifile["DT"]
            if self.inifile["FRAME"] == "F":
                self.frame = "F"
                self.omegaframe = self.inifile["OMEGAFRAME"]
            else:
                self.omegaframe = None
                self.frame = None
        elif self.code == "fargo-adsg":
            self.vtk = self.inifile["Ninterm"] * self.inifile["DT"]
            if self.inifile["Frame"] == "F":
                self.omegaframe = self.inifile["OmegaFrame"]
                self.frame = None
            else:
                self.omegaframe = None
                self.frame = None

    def loadPlanetFile(self, *, planet_number: int = 0):
        planet_file = f"planet{planet_number}.dat"

        if self.code in ("idefix", "fargo3d"):
            if Path(self.directory).joinpath(planet_file).is_file():
                columns = np.loadtxt(os.path.join(self.directory, planet_file)).T
                self.qpl = columns[7]
                self.dpl = np.sqrt(np.sum(columns[1:4] ** 2, axis=0))
                if self.code == "fargo3d" and self.inifile["FRAME"] == "C":
                    self.omegaframe = np.sqrt((1.0 + self.qpl) / pow(self.dpl, 3.0))
                    self.frame = "C"
                # TODO: change (x,y,z) and (vx,vy,vz) when rotation
                self.dtpl = columns[0]
                self.xpl = columns[1]
                self.ypl = columns[2]
                self.zpl = columns[3]
                self.vxpl = columns[4]
                self.vypl = columns[5]
                self.vzpl = columns[6]
                self.tpl = columns[8]
            else:
                raise FileNotFoundError(f"{planet_file} not found")
        elif self.code in ("fargo-adsg"):
            if Path(self.directory).joinpath(planet_file).is_file():
                columns = np.loadtxt(os.path.join(self.directory, planet_file)).T
                self.qpl = columns[5]
                self.dpl = np.sqrt(np.sum(columns[1:3] ** 2, axis=0))
                if self.inifile["Frame"] == "C":
                    self.omegaframe = np.sqrt((1.0 + self.qpl) / pow(self.dpl, 3.0))
                    self.frame = "C"
                # TODO: change (x,y,z) and (vx,vy,vz) when rotation
                self.dtpl = columns[0]
                self.xpl = columns[1]
                self.ypl = columns[2]
                self.zpl = 0.0
                self.vxpl = columns[3]
                self.vypl = columns[4]
                self.vzpl = 0.0
                self.tpl = columns[7]
            else:
                raise FileNotFoundError(f"{planet_file} not found")
        else:
            raise NotImplementedError(
                f"{planet_file} not found for {self.code}. For now, you can't rotate the grid with the planet."
            )

        if self.code in ("idefix", "fargo3d", "fargo-adsg"):
            if self.omegaframe is None or self.frame == "C":
                hx = self.ypl * self.vzpl - self.zpl * self.vypl
                hy = self.zpl * self.vxpl - self.xpl * self.vzpl
                hz = self.xpl * self.vypl - self.ypl * self.vxpl
                hhor = np.sqrt(hx * hx + hy * hy)

                h2 = hx * hx + hy * hy + hz * hz
                h = np.sqrt(h2)
                self.ipl = np.arcsin(hhor / h)

                Ax = (
                    self.vypl * hz
                    - self.vzpl * hy
                    - (1.0 + self.qpl) * self.xpl / self.dpl
                )
                Ay = (
                    self.vzpl * hx
                    - self.vxpl * hz
                    - (1.0 + self.qpl) * self.ypl / self.dpl
                )
                Az = (
                    self.vxpl * hy
                    - self.vypl * hx
                    - (1.0 + self.qpl) * self.zpl / self.dpl
                )

                self.epl = np.sqrt(Ax * Ax + Ay * Ay + Az * Az) / (1.0 + self.qpl)
                self.apl = h * h / ((1.0 + self.qpl) * (1.0 - self.epl * self.epl))
            # else:
            #     raise NotImplementedError(
            #         "We do not yet compute eccentricity, inclination and semi-major axis if fixed rotating frame."
            #     )

    def countSimuFiles(self):
        if self.code in ("fargo3d", "fargo-adsg"):
            self.data_files = [
                fn
                for fn in glob.glob1(self.directory, "gasdens*.dat")
                if re.match(r"gasdens\d+.dat", fn)
            ]
        elif self.code in ("idefix", "pluto"):
            self.data_files = list(glob.glob1(self.directory, "data.*.vtk"))
        else:
            raise RuntimeError("Unknown file format")

    def loadSimuFile(
        self,
        on: int,
        *,
        geometry: str = "unknown",
        cell: str = "edges",
        pattern=None,
    ):
        if self.code == "fargo3d":
            return _load_fargo3d(
                on,
                directory=self.directory,
                inifile=self.paramfile,
                pattern=pattern,
            )
        elif self.code == "fargo-adsg":
            return _load_fargo_adsg(
                on,
                directory=self.directory,
                pattern=pattern,
            )
        elif self.code in ("idefix", "pluto"):
            return _load_idefix(
                on,
                directory=self.directory,
                geometry=geometry,
                cell=cell,
                computedata=True,
                pattern=pattern,
            )
        else:
            raise ValueError(f"For now, can't read files from {self.code} simulations.")


def _load_idefix(
    on: int,
    *,
    directory="",
    geometry="unknown",
    cell="edges",
    computedata=True,
    pattern=None,
) -> gpgi.types.Dataset:
    """
    Adapted from Geoffroy Lesur
    Function that reads a vtk file
    pattern can be a lambda function like
    lambda on:f"data.{on:04d}.vtk"
    """

    directory = os.fspath(directory)
    if pattern is None:
        filename = os.path.join(directory, f"data.{on:04d}.vtk")
    else:
        filename = pattern(on)

    if not os.path.isfile(filename):
        raise FileNotFoundError("Idefix: %s not found." % filename)

    fid = open(filename, "rb")

    # raw data which will be read from the file
    data = {}

    # initialize geometry
    if geometry not in ("unknown", "cartesian", "polar", "spherical", "cylindrical"):
        raise ValueError(f"Unknown geometry value: {geometry  !r}")

    # datatype we read
    dt = np.dtype(">f4")  # Big endian single precision floats
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
                np.fromfile(fid, dt, 1)
            elif entry == "GEOMETRY":
                g = np.fromfile(fid, dint, 1)
                if g == 0:
                    thisgeometry = "cartesian"
                elif g == 1:
                    thisgeometry = "polar"
                elif g == 2:
                    thisgeometry = "spherical"
                elif g == 3:
                    thisgeometry = "cylindrical"
                else:
                    raise ValueError(
                        f"Unknown value for GEOMETRY flag ('{g}') was found in the VTK file."
                    )

                if geometry != "unknown":
                    # We already have a proposed geometry, check that what is read from the file matches
                    if thisgeometry != geometry:
                        raise ValueError(
                            f"geometry argument ('{geometry}') is inconsistent with GEOMETRY flag from the VTK file ('{thisgeometry}')"
                        )
                geometry = thisgeometry
            elif entry == "PERIODICITY":
                tuple(np.fromfile(fid, dint, 3).astype(bool))
            else:
                raise ValueError(f"Received unknown field: '{entry}'.")

            s = fid.readline()  # extra linefeed

        # finished reading the field entry
        # read next line
        s = fid.readline()  # DIMENSIONS...

    if geometry == "unknown":
        raise RuntimeError(
            "Geometry couldn't be determined from data. "
            "Try to set the geometry keyword argument explicitely."
        )

    slist = s.split()  # DIMENSIONS....
    n1 = int(slist[1])
    n2 = int(slist[2])
    n3 = int(slist[3])

    x: _AnyArray
    y: _AnyArray
    z: _AnyArray

    if geometry in ("cartesian", "cylindrical"):
        # CARTESIAN geometry
        # NOTE: cylindrical geometry is meant to be only used in 2D
        #       so the expected coordinates (R, z) are never curvilinear,
        #       which means we can treat them as cartesian
        s = fid.readline()  # X_COORDINATES NX float
        inipos = (
            fid.tell()
        )  # we store the file pointer position before computing points
        logger.debug("loading the X-grid cells: {}", n1)
        x = np.memmap(
            fid, mode="r", dtype=dt, offset=inipos, shape=n1
        )  # some smart memory efficient way to store the array
        newpos = (
            np.float32().nbytes * 1 * n1 + inipos
        )  # we calculate the offset that we would expect normally with a np.fromfile
        fid.seek(newpos, os.SEEK_SET)  # we set the file pointer position to this offset
        s = fid.readline()  # Extra line feed added by idefix

        s = fid.readline()  # Y_COORDINATES NY float
        inipos = (
            fid.tell()
        )  # we store the file pointer position before computing points
        logger.debug("loading the Y-grid cells: {}", n2)
        y = np.memmap(
            fid, mode="r", dtype=dt, offset=inipos, shape=n2
        )  # some smart memory efficient way to store the array
        newpos = (
            np.float32().nbytes * 1 * n2 + inipos
        )  # we calculate the offset that we would expect normally with a np.fromfile
        fid.seek(newpos, os.SEEK_SET)  # we set the file pointer position to this offset
        s = fid.readline()  # Extra line feed added by idefix

        s = fid.readline()  # Z_COORDINATES NZ float
        inipos = (
            fid.tell()
        )  # we store the file pointer position before computing points
        logger.debug("loading the Z-grid cells: {}", n3)
        z = np.memmap(
            fid, mode="r", dtype=dt, offset=inipos, shape=n3
        )  # some smart memory efficient way to store the array
        newpos = (
            np.float32().nbytes * 1 * n3 + inipos
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
            if n1 > 1:
                n1 = n1 - 1
                if cell == "centers":
                    x1 = 0.5 * (x[1:] + x[:-1])
                elif cell == "edges":
                    x1 = x
            else:
                x1 = x
            if n2 > 1:
                n2 = n2 - 1
                if cell == "centers":
                    x2 = 0.5 * (y[1:] + y[:-1])
                elif cell == "edges":
                    x2 = y
            else:
                x2 = y
            if n3 > 1:
                n3 = n3 - 1
                if cell == "centers":
                    x3 = 0.5 * (z[1:] + z[:-1])
                elif cell == "edges":
                    x3 = z
            else:
                x3 = z
        elif point_type == "POINT_DATA":
            x1 = x
            x2 = y
            x3 = z

        if n1 * n2 * n3 != npoints:
            raise ValueError(
                "Idefix: Grid size (%d) incompatible with number of points (%d) in the data set"
                % (n1 * n2 * n3, npoints)
            )

    else:
        # POLAR or SPHERICAL coordinates
        if n3 == 1:
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
        logger.debug("loading grid (shape = ({},{},{}))", n1, n2, n3)
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

        # points=points
        if n1 * n2 * n3 != npoints:
            raise ValueError(
                "Idefix: Grid size (%d) incompatible with number of points (%d) in the data set"
                % (n1 * n2 * n3, npoints)
            )

        # Reconstruct the polar coordinate system
        x1d = points[::3]
        y1d = points[1::3]
        z1d = points[2::3]

        xcart = np.transpose(x1d.reshape(n3, n2, n1))
        ycart = np.transpose(y1d.reshape(n3, n2, n1))
        zcart = np.transpose(z1d.reshape(n3, n2, n1))

        # Reconstruct the polar coordinate system
        if geometry == "polar":
            r = np.sqrt(xcart[:, 0, 0] ** 2 + ycart[:, 0, 0] ** 2)
            theta = np.unwrap(np.arctan2(ycart[0, :, 0], xcart[0, :, 0]))
            z = zcart[0, 0, :]
            s = fid.readline()  # CELL_DATA (NX-1)(NY-1)(NZ-1)
            slist = s.split()
            data_type = str(slist[0], "utf-8")
            if data_type != "CELL_DATA":
                fid.close()
                raise ValueError(
                    "Idefix: this routine expect 'CELL DATA' as produced by PLUTO, not '%s'."
                    % data_type
                )
            s = fid.readline()  # Line feed

            if cell == "edges":
                if n1 > 1:
                    n1 = n1 - 1
                    x1 = r
                else:
                    x1 = r
                if n2 > 1:
                    n2 = n2 - 1
                    x2 = theta
                else:
                    x2 = theta
                if n3 > 1:
                    n3 = n3 - 1
                    x3 = z
                else:
                    x3 = z

            # Perform averaging on coordinate system to get cell centers
            # The file contains face coordinates, so we extrapolate to get the cell center coordinates.
            elif cell == "centers":
                if n1 > 1:
                    n1 = n1 - 1
                    x1 = 0.5 * (r[1:] + r[:-1])
                else:
                    x1 = r
                if n2 > 1:
                    n2 = n2 - 1
                    x2 = (0.5 * (theta[1:] + theta[:-1]) + np.pi) % (
                        2.0 * np.pi
                    ) - np.pi
                else:
                    x2 = theta
                if n3 > 1:
                    n3 = n3 - 1
                    x3 = 0.5 * (z[1:] + z[:-1])
                else:
                    x3 = z

        # Reconstruct the spherical coordinate system
        if geometry == "spherical":
            if is2d:
                r = np.sqrt(xcart[:, 0, 0] ** 2 + ycart[:, 0, 0] ** 2)
                phi = np.unwrap(np.arctan2(zcart[0, n2 // 2, :], xcart[0, n2 // 2, :]))
                theta = np.arccos(
                    ycart[0, :, 0] / np.sqrt(xcart[0, :, 0] ** 2 + ycart[0, :, 0] ** 2)
                )
            else:
                r = np.sqrt(
                    xcart[:, 0, 0] ** 2 + ycart[:, 0, 0] ** 2 + zcart[:, 0, 0] ** 2
                )
                phi = np.unwrap(
                    np.arctan2(
                        ycart[n1 // 2, n2 // 2, :],
                        xcart[n1 // 2, n2 // 2, :],
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
                    "Idefix: this routine expect 'CELL DATA' as produced by PLUTO, not '%s'."
                    % data_type
                )
            s = fid.readline()  # Line feed

            if cell == "edges":
                if n1 > 1:
                    n1 = n1 - 1
                    x1 = r
                else:
                    x1 = r
                if n2 > 1:
                    n2 = n2 - 1
                    x2 = theta
                else:
                    x2 = theta
                if n3 > 1:
                    n3 = n3 - 1
                    x3 = phi
                else:
                    x3 = phi

            # Perform averaging on coordinate system to get cell centers
            # The file contains face coordinates, so we extrapolate to get the cell center coordinates.
            elif cell == "centers":
                if n1 > 1:
                    n1 = n1 - 1
                    x1 = 0.5 * (r[1:] + r[:-1])
                else:
                    x1 = r
                if n2 > 1:
                    n2 = n2 - 1
                    x2 = 0.5 * (theta[1:] + theta[:-1])
                else:
                    x2 = theta
                if n3 > 1:
                    n3 = n3 - 1
                    x3 = 0.5 * (phi[1:] + phi[:-1])
                else:
                    x3 = phi

    if computedata:
        logger.debug("loading data arrays")
        while 1:
            s = (
                fid.readline()
            )  # SCALARS/VECTORS name data_type (ex: SCALARS imagedata unsigned_char)
            # print repr(s)
            if len(s) < 2:  # leave if end of file
                break
            slist = s.split()
            datatype = str(slist[0], "utf-8")
            varname = str(slist[1], "utf-8").upper()
            if datatype == "SCALARS":
                fid.readline()  # LOOKUP TABLE

                inipos = (
                    fid.tell()
                )  # we store the file pointer position before computing points
                # array = np.fromfile(fid, dt, n1 * n2 * n3).reshape(n3, n2, n1)
                array = np.memmap(
                    fid,
                    mode="r",
                    dtype=dt,
                    offset=inipos,
                    shape=n1 * n2 * n3,
                ).reshape(
                    n3, n2, n1
                )  # some smart memory efficient way to store the array
                newpos = (
                    np.float32().nbytes * n1 * n2 * n3 + inipos
                )  # we calculate the offset that we would expect normally with a np.fromfile
                fid.seek(
                    newpos, os.SEEK_SET
                )  # we set the file pointer position to this offset

                data[varname] = np.transpose(array)
            elif datatype == "VECTORS":
                inipos = (
                    fid.tell()
                )  # we store the file pointer position before computing points
                Q = np.memmap(
                    fid,
                    mode="r",
                    dtype=dt,
                    offset=inipos,
                    shape=n1 * n2 * n3,
                )  # some smart memory efficient way to store the array
                # Q = np.fromfile(fid, dt, 3 * n1 * n2 * n3)
                newpos = (
                    np.float32().nbytes * n1 * n2 * n3 + inipos
                )  # we calculate the offset that we would expect normally with a np.fromfile
                fid.seek(
                    newpos, os.SEEK_SET
                )  # we set the file pointer position to this offset

                data[varname + "_X"] = np.transpose(Q[::3].reshape(n3, n2, n1))
                data[varname + "_Y"] = np.transpose(Q[1::3].reshape(n3, n2, n1))
                data[varname + "_Z"] = np.transpose(Q[2::3].reshape(n3, n2, n1))

            else:
                raise ValueError(
                    "Idefix: Unknown datatype '%s', should be 'SCALARS' or 'VECTORS'"
                    % datatype
                )

            logger.debug("read field: {}", varname)

            fid.readline()  # extra line feed
    fid.close()
    if x1.shape[0] == 1:
        x1 = np.array([x1[0], x1[0]])
    if x2.shape[0] == 1:
        x2 = np.array([x2[0], x2[0]])
    if x3.shape[0] == 1:
        x3 = np.array([x3[0], x3[0]])

    axes = {
        "cartesian": ("x", "y", "z"),
        "polar": ("radius", "azimuth", "z"),
        "cylindrical": ("azimuth", "radius", "z"),
        "spherical": ("radius", "colatitude", "azimuth"),
    }[geometry]
    cell_edges = {axis: arr.astype(dt) for axis, arr in zip(axes, (x1, x2, x3))}

    return gpgi.load(
        geometry=geometry,
        grid={
            "fields": data,
            "cell_edges": cell_edges,
        },
        metadata={"code": "idefix"},
    )


def _load_fargo3d(
    on: int, *, directory="", inifile="", pattern=None
) -> gpgi.types.Dataset:
    """
    pattern can be the type of dust fluid given in the file name
    """
    directory = Path(directory)

    if pattern is None:
        densfile = directory / f"gasdens{on}.dat"
        vyfile = directory / f"gasvy{on}.dat"
        vxfile = directory / f"gasvx{on}.dat"
        vzfile = directory / f"gasvz{on}.dat"
    else:
        densfile = directory / f"{pattern}dens{on}.dat"
        vyfile = directory / f"{pattern}vy{on}.dat"
        vxfile = directory / f"{pattern}vx{on}.dat"
        vzfile = directory / f"{pattern}vz{on}.dat"

    params = Parameters(
        directory=directory, inifile=inifile, code="" if inifile == "" else "fargo3d"
    )
    params.loadIniFile()

    geometry = params.inifile["COORDINATES"]
    data = {}

    domain_x = np.loadtxt(directory / "domain_x.dat")
    DTYPE = domain_x.dtype
    # We avoid ghost cells
    domain_y = np.loadtxt(directory / "domain_y.dat").astype(DTYPE)[3:-3]
    domain_z = np.loadtxt(directory / "domain_z.dat").astype(DTYPE)
    if domain_z.shape[0] > 6:
        domain_z = domain_z[3:-3]

    if geometry == "cylindrical":
        x1 = domain_y  # X-Edge
        x2 = domain_x  # Y-Edge
        x3 = domain_z  # Z-Edge #latitute

        n1 = len(x1) - 1  # if len(x1)>2 else 2
        n2 = len(x2) - 1  # if len(x2)>2 else 2
        n3 = len(x3) - 1  # if len(x3)>2 else 2
        if densfile.is_file():
            data["RHO"] = (
                np.fromfile(densfile, dtype=DTYPE)
                .reshape(n3, n1, n2)
                .transpose(1, 2, 0)
            )  # rad, pĥi, z
        if vyfile.is_file():
            data["VX1"] = (
                np.fromfile(vyfile, dtype=DTYPE).reshape(n3, n1, n2).transpose(1, 2, 0)
            )  # rad, pĥi, z
        if vxfile.is_file():
            data["VX2"] = (
                np.fromfile(vxfile, dtype=DTYPE).reshape(n3, n1, n2).transpose(1, 2, 0)
            )  # rad, pĥi, z
        if vzfile.is_file():
            data["VX3"] = (
                np.fromfile(vzfile, dtype=DTYPE).reshape(n3, n1, n2).transpose(1, 2, 0)
            )  # rad, pĥi, z
        for key in list(data.keys()):
            data[key] = np.roll(data[key], n2 // 2, axis=1)
        if x1.shape[0] == 1:
            x1 = np.array([x1[0], x1[0]])
        if x2.shape[0] == 1:
            x2 = np.array([x2[0], x2[0]])
        if x3.shape[0] == 1:
            x3 = np.array([x3[0], x3[0]])
        return gpgi.load(
            geometry="polar",
            grid={
                "fields": data,
                "cell_edges": {
                    "radius": x1,
                    "azimuth": x2 + np.pi,
                    "z": x3,
                },
            },
            metadata={"code": "fargo3d"},
        )
    elif geometry == "spherical":
        x1 = domain_y  # X-Edge
        x2 = domain_z  # Z-Edge #latitute
        x3 = domain_x  # Y-Edge

        n1 = len(x1) - 1  # if len(x1)>2 else 2
        n2 = len(x2) - 1  # if len(x2)>2 else 2
        n3 = len(x3) - 1  # if len(x3)>2 else 2
        if densfile.is_file():
            data["RHO"] = (
                np.fromfile(densfile, dtype=DTYPE)
                .reshape(n2, n1, n3)
                .transpose(1, 0, 2)
            )  # rad, pĥi, z
        if vyfile.is_file():
            data["VX1"] = (
                np.fromfile(vyfile, dtype=DTYPE).reshape(n2, n1, n3).transpose(1, 0, 2)
            )  # rad, pĥi, z
        if vzfile.is_file():
            data["VX2"] = (
                np.fromfile(vzfile, dtype=DTYPE).reshape(n2, n1, n3).transpose(1, 0, 2)
            )  # rad, pĥi, z
        if vxfile.is_file():
            data["VX3"] = (
                np.fromfile(vxfile, dtype=DTYPE).reshape(n2, n1, n3).transpose(1, 0, 2)
            )  # rad, pĥi, z
        for key in list(data.keys()):
            data[key] = np.roll(data[key], n2 // 2, axis=2)
        if x1.shape[0] == 1:
            x1 = np.array([x1[0], x1[0]])
        if x2.shape[0] == 1:
            x2 = np.array([x2[0], x2[0]])
        if x3.shape[0] == 1:
            x3 = np.array([x3[0], x3[0]])
        return gpgi.load(
            geometry=geometry,
            grid={
                "fields": data,
                "cell_edges": {
                    "radius": x1,
                    "colatitude": x2,
                    "azimuth": x3 + np.pi,
                },
            },
            metadata={"code": "fargo3d"},
        )
    else:
        raise ValueError(f"{geometry} not implemented yet for fargo3d.")


def _load_fargo_adsg(on: int, *, directory="", pattern=None) -> gpgi.types.Dataset:
    """
    pattern can be gas (default) or dust
    """
    directory = Path(directory)

    if pattern is None:
        densfile = directory / f"gasdens{on}.dat"
        vyfile = directory / f"gasvrad{on}.dat"
        vxfile = directory / f"gasvtheta{on}.dat"
    else:
        pattern(on)
        raise NotImplementedError("pattern not implemented yet (reading dust files)")

    data = {}

    phi = np.loadtxt(directory / "used_azi.dat")[:, 0]

    DTYPE = phi.dtype

    domain_x = np.zeros(len(phi) + 1, dtype=DTYPE)
    domain_x[:-1] = phi
    domain_x[-1] = 2 * np.pi
    domain_x -= np.pi
    # We avoid ghost cells
    domain_y = np.loadtxt(directory / "used_rad.dat").astype(DTYPE)
    domain_z = np.zeros(2, dtype=DTYPE)

    x1 = domain_y  # X-Edge
    x2 = domain_x  # Y-Edge
    x3 = domain_z  # Z-Edge #latitute

    n1 = len(x1) - 1  # if len(x1)>2 else 2
    n2 = len(x2) - 1  # if len(x2)>2 else 2
    n3 = len(x3) - 1  # if len(x3)>2 else 2
    if densfile.is_file():
        data["RHO"] = (
            np.fromfile(densfile, dtype=DTYPE).reshape(n3, n1, n2).transpose(1, 2, 0)
        )  # rad, pĥi, z
    if vyfile.is_file():
        data["VX1"] = (
            np.fromfile(vyfile, dtype=DTYPE).reshape(n3, n1, n2).transpose(1, 2, 0)
        )  # rad, pĥi, z
    if vxfile.is_file():
        data["VX2"] = (
            np.fromfile(vxfile, dtype=DTYPE).reshape(n3, n1, n2).transpose(1, 2, 0)
        )  # rad, pĥi, z
    if x1.shape[0] == 1:
        x1 = np.array([x1[0], x1[0]])
    if x2.shape[0] == 1:
        x2 = np.array([x2[0], x2[0]])
    if x3.shape[0] == 1:
        x3 = np.array([x3[0], x3[0]])
    return gpgi.load(
        geometry="polar",
        grid={
            "fields": data,
            "cell_edges": {
                "radius": x1,
                "azimuth": x2 + np.pi,
                "z": x3,
            },
        },
        metadata={"code": "fargo-adsg"},
    )
