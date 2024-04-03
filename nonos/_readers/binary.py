__all__ = [
    "VTKReader",
    "Fargo3DReader",
    "FargoADSGReader",
]
import json
import os
import re
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, final

import numpy as np

from nonos._readers._base import ReaderMixin
from nonos._types import BinData, FloatArray, Geometry, PathT

if sys.version_info >= (3, 11):
    from typing import assert_never
else:
    from typing_extensions import assert_never

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


@final
class VTKReader(ReaderMixin):
    @staticmethod
    def parse_output_number_and_filename(
        file_or_number: Union[PathT, int],
        *,
        directory: PathT,
        prefix: str,  # noqa: ARG004
    ) -> Tuple[int, Path]:
        if isinstance(file_or_number, (str, Path)):
            file = Path(file_or_number)
            if (m := re.search(r"\d+", file.name)) is None:
                raise ValueError(
                    f"Failed to parse output number from filename {file.name}"
                )
            else:
                output_number = int(m.group())
        else:
            output_number = int(file_or_number)
            file = Path(f"data.{output_number:04d}.vtk")

        if file == Path(file.name):
            file = Path(directory) / file

        return output_number, file

    @staticmethod
    def get_bin_files(directory: PathT, /) -> List[Path]:
        directory = Path(directory)
        return sorted(directory.glob("data.*.vtk"))

    @staticmethod
    def read(file, /, **meta) -> BinData:
        """
        Adapted from Geoffroy Lesur
        Function that reads a vtk file in polar coordinates
        """

        meta.setdefault("geometry", "unset")
        meta.setdefault("cell", "edges")
        meta.setdefault("computedata", True)

        fid = open(file, "rb")

        # define our datastructure
        V = BinData.default_init()

        # raw data which will be read from the file
        V["data"] = {}

        # initialize geometry
        V["geometry"] = Geometry(meta["geometry"])

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
                    # skip
                    fid.seek(dt.itemsize, os.SEEK_CUR)
                elif entry == "GEOMETRY":
                    g = np.fromfile(fid, dint, 1)
                    if g == 0:
                        thisgeometry = Geometry.CARTESIAN
                    elif g == 1:
                        thisgeometry = Geometry.POLAR
                    elif g == 2:
                        thisgeometry = Geometry.SPHERICAL
                    elif g == 3:
                        thisgeometry = Geometry.CYLINDRICAL
                    else:  # pragma: no cover
                        fid.close()
                        raise ValueError(
                            f"Unknown value for GEOMETRY flag ({g!r}) was found in the VTK file."
                        )

                    if V["geometry"] is not Geometry.UNSET:
                        # We already have a proposed geometry, check that what is read from the file matches
                        if thisgeometry != V["geometry"]:  # pragma: no cover
                            fid.close()
                            raise ValueError(
                                f"geometry argument ({V['geometry']!r}) is "
                                "inconsistent with GEOMETRY flag from the VTK file "
                                f"({thisgeometry!r})"
                            )
                    V["geometry"] = thisgeometry
                elif entry == "PERIODICITY":
                    # skip
                    fid.seek(dint.itemsize * 3, os.SEEK_CUR)
                else:  # pragma: no cover
                    fid.close()
                    raise ValueError(f"Received unknown field: {entry!r}")

                s = fid.readline()  # extra linefeed

            # finished reading the field entry
            # read next line
            s = fid.readline()  # DIMENSIONS...

        if V["geometry"] is Geometry.UNSET:  # pragma: no cover
            fid.close()
            raise RuntimeError(
                "Geometry couldn't be determined from data. "
                "Try to set the geometry keyword argument explicitly."
            )
        slist = s.split()  # DIMENSIONS....
        n1 = int(slist[1])
        n2 = int(slist[2])
        n3 = int(slist[3])

        z: Union["np.ndarray", "np.memmap"]

        if V["geometry"] is Geometry.CARTESIAN:
            s = fid.readline()  # X_COORDINATES NX float
            # we store the file pointer position before computing points
            inipos = fid.tell()
            # some smart memory efficient way to store the array
            x = np.memmap(fid, mode="r", dtype=dt, offset=inipos, shape=n1)
            # we calculate the offset that we would expect normally with a np.fromfile
            newpos = dt.itemsize * n1 + inipos
            # we set the file pointer position to this offset
            fid.seek(newpos, os.SEEK_SET)
            s = fid.readline()  # Extra line feed added by idefix

            s = fid.readline()  # Y_COORDINATES NY float
            # we store the file pointer position before computing points
            inipos = fid.tell()
            # some smart memory efficient way to store the array
            y = np.memmap(fid, mode="r", dtype=dt, offset=inipos, shape=n2)
            # we calculate the offset that we would expect normally with a np.fromfile
            newpos = dt.itemsize * n2 + inipos
            # we set the file pointer position to this offset
            fid.seek(newpos, os.SEEK_SET)
            s = fid.readline()  # Extra line feed added by idefix

            s = fid.readline()  # Z_COORDINATES NZ float
            # we store the file pointer position before computing points
            inipos = fid.tell()
            # some smart memory efficient way to store the array
            z = np.memmap(fid, mode="r", dtype=dt, offset=inipos, shape=n3)
            # we calculate the offset that we would expect normally with a np.fromfile
            newpos = dt.itemsize * 1 * n3 + inipos
            # we set the file pointer position to this offset
            fid.seek(newpos, os.SEEK_SET)
            s = fid.readline()  # Extra line feed added by idefix

            s = fid.readline()  # POINT_DATA NXNYNZ

            slist = s.split()
            point_type = str(slist[0], "utf-8")
            npoints = int(slist[1])
            s = fid.readline()  # EXTRA LINE FEED

            if point_type == "CELL_DATA" or meta["cell"] == "centers":
                # The file contains face coordinates, so we extrapolate to get the cell center coordinates.
                if n1 > 1:
                    n1 -= 1
                    if meta["cell"] == "centers":
                        V["x1"] = 0.5 * (x[1:] + x[:-1])
                    elif meta["cell"] == "edges":
                        V["x1"] = x
                else:
                    V["x1"] = x
                if n2 > 1:
                    n2 -= 1
                    if meta["cell"] == "centers":
                        V["x2"] = 0.5 * (y[1:] + y[:-1])
                    elif meta["cell"] == "edges":
                        V["x2"] = y
                else:
                    V["x2"] = y
                if n3 > 1:
                    n3 -= 1
                    if meta["cell"] == "centers":
                        V["x3"] = 0.5 * (z[1:] + z[:-1])
                    elif meta["cell"] == "edges":
                        V["x3"] = z
                else:
                    V["x3"] = z
            elif point_type == "POINT_DATA":
                V["x1"] = x
                V["x2"] = y
                V["x3"] = z

            if (grid_size := n1 * n2 * n3) != npoints:  # pragma: no cover
                fid.close()
                raise ValueError(
                    f"Grid size ({grid_size}) is not consistent with number "
                    f"of points ({npoints}) in the data set"
                )
            del grid_size

        elif V["geometry"] is Geometry.POLAR or V["geometry"] is Geometry.SPHERICAL:
            if n3 == 1:
                is2d = 1
            else:
                is2d = 0

            s = fid.readline()  # POINTS NXNYNZ float
            slist = s.split()
            npoints = int(slist[1])

            # we store the file pointer position before computing points
            inipos = fid.tell()

            # some smart memory efficient way to store the array
            points = np.memmap(
                fid, mode="r", dtype=dt, offset=inipos, shape=3 * npoints
            )
            # we calculate the offset that we would expect normally with a np.fromfile
            newpos = dt.itemsize * 3 * npoints + inipos
            # we set the file pointer position to this offset
            fid.seek(newpos, os.SEEK_SET)
            s = fid.readline()  # EXTRA LINE FEED

            if (grid_size := n1 * n2 * n3) != npoints:  # pragma: no cover
                fid.close()
                raise ValueError(
                    f"Grid size ({grid_size}) is not consistent with number "
                    f"of points ({npoints}) in the data set"
                )
            del grid_size

            # Reconstruct the polar coordinate system
            x1d = points[::3]
            y1d = points[1::3]
            z1d = points[2::3]

            new_shape = n3, n2, n1
            xcart = np.transpose(x1d.reshape(new_shape))
            ycart = np.transpose(y1d.reshape(new_shape))
            zcart = np.transpose(z1d.reshape(new_shape))
            del new_shape

            # Reconstruct the polar coordinate system
            if V["geometry"] is Geometry.POLAR:
                r = np.sqrt(xcart[:, 0, 0] ** 2 + ycart[:, 0, 0] ** 2)
                theta = np.unwrap(np.arctan2(ycart[0, :, 0], xcart[0, :, 0]))
                z = zcart[0, 0, :]

                s = fid.readline()  # CELL_DATA (NX-1)(NY-1)(NZ-1)
                slist = s.split()
                data_type = str(slist[0], "utf-8")
                if data_type != "CELL_DATA":  # pragma: no cover
                    fid.close()
                    raise ValueError(
                        "Expected 'CELL DATA' as produced by PLUTO, got {data_type!r}."
                    )
                s = fid.readline()  # Line feed

                if meta["cell"] == "edges":
                    if n1 > 1:
                        n1 -= 1
                        V["x1"] = r
                    else:
                        V["x1"] = r
                    if n2 > 1:
                        n2 -= 1
                        V["x2"] = theta
                    else:
                        V["x2"] = theta
                    if n3 > 1:
                        n3 -= 1
                        V["x3"] = z
                    else:
                        V["x3"] = z

                # Perform averaging on coordinate system to get cell centers
                # The file contains face coordinates, so we extrapolate to get the cell center coordinates.
                elif meta["cell"] == "centers":
                    if n1 > 1:
                        n1 -= 1
                        V["x1"] = 0.5 * (r[1:] + r[:-1])
                    else:
                        V["x1"] = r
                    if n2 > 1:
                        n2 -= 1
                        V["x2"] = (0.5 * (theta[1:] + theta[:-1]) + np.pi) % (
                            2.0 * np.pi
                        ) - np.pi
                    else:
                        V["x2"] = theta
                    if n3 > 1:
                        n3 -= 1
                        V["x3"] = 0.5 * (z[1:] + z[:-1])
                    else:
                        V["x3"] = z

            # Reconstruct the spherical coordinate system
            elif V["geometry"] is Geometry.SPHERICAL:
                if is2d:
                    r = np.sqrt(xcart[:, 0, 0] ** 2 + ycart[:, 0, 0] ** 2)
                    phi = np.unwrap(
                        np.arctan2(zcart[0, n2 // 2, :], xcart[0, n2 // 2, :])
                    )
                    theta = np.arccos(
                        ycart[0, :, 0]
                        / np.sqrt(xcart[0, :, 0] ** 2 + ycart[0, :, 0] ** 2)
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
                            xcart[0, :, 0] ** 2
                            + ycart[0, :, 0] ** 2
                            + zcart[0, :, 0] ** 2
                        )
                    )

                s = fid.readline()  # CELL_DATA (NX-1)(NY-1)(NZ-1)
                slist = s.split()
                data_type = str(slist[0], "utf-8")
                if data_type != "CELL_DATA":  # pragma: no cover
                    fid.close()
                    raise ValueError(
                        f"Expected 'CELL DATA' as produced by PLUTO. "
                        f"Got {data_type!r}"
                    )
                s = fid.readline()  # Line feed

                if meta["cell"] == "edges":
                    if n1 > 1:
                        n1 -= 1
                        V["x1"] = r
                    else:
                        V["x1"] = r
                    if n2 > 1:
                        n2 -= 1
                        V["x2"] = theta
                    else:
                        V["x2"] = theta
                    if n3 > 1:
                        n3 -= 1
                        V["x3"] = phi
                    else:
                        V["x3"] = phi

                # Perform averaging on coordinate system to get cell centers
                # The file contains face coordinates, so we extrapolate to get the cell center coordinates.
                elif meta["cell"] == "centers":
                    if n1 > 1:
                        n1 -= 1
                        V["x1"] = 0.5 * (r[1:] + r[:-1])
                    else:
                        V["x1"] = r
                    if n2 > 1:
                        n2 -= 1
                        V["x2"] = 0.5 * (theta[1:] + theta[:-1])
                    else:
                        V["x2"] = theta
                    if n3 > 1:
                        n3 -= 1
                        V["x3"] = 0.5 * (phi[1:] + phi[:-1])
                    else:
                        V["x3"] = phi
            else:
                assert_never(V["geometry"])
        else:
            assert_never(V["geometry"])

        if meta["computedata"]:
            new_shape = n3, n2, n1
            grid_size = int(np.prod(new_shape))
            while 1:
                # SCALARS/VECTORS name data_type (ex: SCALARS imagedata unsigned_char)
                s = fid.readline()
                if len(s) < 2:  # leave if end of file
                    break
                slist = s.split()
                datatype = str(slist[0], "utf-8")
                varname = str(slist[1], "utf-8").upper()
                if datatype == "SCALARS":
                    fid.readline()  # LOOKUP TABLE

                    # we store the file pointer position before computing points
                    inipos = fid.tell()

                    # some smart memory efficient way to store the array
                    array = np.memmap(
                        fid, mode="r", dtype=dt, offset=inipos, shape=grid_size
                    ).reshape(new_shape)
                    # we calculate the offset that we would expect normally with a np.fromfile
                    newpos = inipos + dt.itemsize * grid_size
                    # we set the file pointer position to this offset
                    fid.seek(newpos, os.SEEK_SET)

                    V["data"][varname] = np.transpose(array)
                elif datatype == "VECTORS":
                    # we store the file pointer position before computing points
                    inipos = fid.tell()
                    nelements = 3 * grid_size
                    # some smart memory efficient way to store the array
                    Q = np.memmap(
                        fid, mode="r", dtype=dt, offset=inipos, shape=nelements
                    )
                    # we calculate the offset that we would expect normally with a np.fromfile
                    newpos = inipos + dt.itemsize * nelements
                    # we set the file pointer position to this offset
                    fid.seek(newpos, os.SEEK_SET)

                    for i, suffix in enumerate("XYZ"):
                        name = f"{varname}_{suffix}"
                        V["data"][name] = np.transpose(Q[i::3].reshape(new_shape))

                else:  # pragma: no cover
                    fid.close()
                    raise ValueError(
                        f"Unknown datatype {datatype!r}. "
                        "Expected 'SCALARS' or 'VECTORS'"
                    )

                fid.readline()  # extra line feed
        fid.close()

        return BinData(**V)


class _FargoReader(ReaderMixin, ABC):
    @staticmethod
    def parse_output_number_and_filename(
        file_or_number: Union[PathT, int],
        *,
        directory: PathT,
        prefix: str,  # noqa ARG004
    ) -> Tuple[int, Path]:
        directory = Path(directory).resolve()
        if isinstance(file_or_number, int):
            output_number = file_or_number
            file = directory / f"gasdens{output_number:04d}.dat"
        else:
            file = Path(file_or_number)
            if file == Path(file.name):
                file = directory / file
            if len(matches := re.findall(r"\d+", file.name)) == 1:
                output_number = int(matches[0])
            elif len(matches) == 0:
                raise RuntimeError(
                    rf"Failed to guess an output number from {file_or_number!r}"
                )
            else:
                raise RuntimeError(rf"Ambiguous output number from {file_or_number!r}")
        return output_number, file

    @staticmethod
    def get_bin_files(directory: PathT, /) -> List[Path]:
        directory = Path(directory)
        return [
            fn
            for fn in sorted(directory.glob("gasdens*.dat"))
            if re.search(r"gasdens\d+.dat$", str(fn)) is not None
        ]

    @staticmethod
    def _get_output_number_and_dir_from(file) -> Tuple[int, Path]:
        _in_file = Path(file).resolve()
        directory = _in_file.parent
        if (match := re.search(r"(?P<on>\d+).dat$", _in_file.name)) is not None:
            output_number = int(match.group("on"))
        else:
            raise ValueError(f"Failed to parse filename {file!r}")

        return output_number, directory

    @staticmethod
    @abstractmethod
    def read(file: PathT, /, **meta) -> BinData: ...


@final
class Fargo3DReader(_FargoReader):
    @override
    @staticmethod
    def read(
        file: PathT,
        /,
        **meta,
    ) -> BinData:
        output_number, directory = _FargoReader._get_output_number_and_dir_from(file)

        default_fluid = "gas"
        fluid_option: Optional[str] = meta.get("fluid", default_fluid)
        if fluid_option is None:
            fluid = default_fluid
        else:
            fluid = fluid_option

        V = BinData.default_init()
        V["geometry"] = Geometry(meta["COORDINATES"])
        V["data"] = {}

        domain_x = np.loadtxt(directory / "domain_x.dat")
        # We avoid ghost cells
        domain_y = np.loadtxt(directory / "domain_y.dat")[3:-3]
        domain_z = np.loadtxt(directory / "domain_z.dat")
        if domain_z.shape[0] > 6:
            domain_z = domain_z[3:-3]

        V["x1"] = domain_y  # X-Edge
        if V["geometry"] is Geometry.CYLINDRICAL:
            V["x2"] = domain_x  # Y-Edge
            V["x3"] = domain_z  # Z-Edge #latitute
        elif V["geometry"] is Geometry.SPHERICAL:
            V["x2"] = domain_z  # Z-Edge #latitute
            V["x3"] = domain_x  # Y-Edge
        n1 = len(V["x1"]) - 1
        n2 = len(V["x2"]) - 1
        n3 = len(V["x3"]) - 1
        grid_shape = n3, n1, n2
        shift = n2 // 2

        if V["geometry"] is Geometry.CYLINDRICAL:
            V["geometry"] = Geometry.POLAR
            pairs = [("RHO", "dens"), ("VX1", "vy"), ("VX2", "vx"), ("VX3", "vz")]
        elif V["geometry"] is Geometry.SPHERICAL:
            pairs = [("RHO", "dens"), ("VX1", "vy"), ("VX2", "vz"), ("VX3", "vx")]
        else:
            raise NotImplementedError

        def _read_array(file: Path):
            return np.roll(
                np.fromfile(file, dtype="float64")
                .reshape(grid_shape)
                .transpose(1, 2, 0),
                shift,
                axis=1,
            )

        for key, field in pairs:
            file = directory / f"{fluid}{field}{output_number}.dat"
            if not file.is_file():
                continue
            V["data"][key] = _read_array(file)

        if not V["data"]:
            raise FileNotFoundError(
                f"No file matches the pattern '{fluid}*{output_number}.dat'"
            )

        return BinData(**V)


@final
class FargoADSGReader(_FargoReader):
    @override
    @staticmethod
    def read(
        file: PathT,
        /,
        **meta,  # noqa: ARG004
    ) -> BinData:
        output_number, directory = _FargoReader._get_output_number_and_dir_from(file)

        V = BinData.default_init()
        V["geometry"] = Geometry.CYLINDRICAL
        V["data"] = {}

        phi = np.loadtxt(directory / "used_azi.dat")[:, 0]
        domain_x = np.zeros(len(phi) + 1)
        domain_x[:-1] = phi
        domain_x[-1] = 2 * np.pi
        domain_x -= np.pi
        # We avoid ghost cells
        domain_y = np.loadtxt(directory / "used_rad.dat")
        domain_z = np.zeros(2)

        V["x1"] = domain_y  # X-Edge
        V["x2"] = domain_x  # Y-Edge
        V["x3"] = domain_z  # Z-Edge #latitute

        n1 = len(V["x1"]) - 1
        n2 = len(V["x2"]) - 1
        n3 = len(V["x3"]) - 1
        grid_shape = n3, n1, n2

        def _read_array(file: Path):
            return (
                np.fromfile(file, dtype="float64")
                .reshape(grid_shape)
                .transpose(1, 2, 0)
            )

        for key, field in [("RHO", "dens"), ("VX1", "vrad"), ("VX2", "vtheta")]:
            file = directory / f"gas{field}{output_number}.dat"
            if not file.is_file():
                continue
            V["data"][key] = _read_array(file)
        return BinData(**V)


class NPYReader(ReaderMixin):

    # we accept a leading '_' for backward compatibility
    _filename_re = re.compile(
        r"^_?(?P<prefix>[\w]*((\d*\.)?\d+)?)"
        r"_(?P<field_name>[A-Z]+)"
        r"\.(?P<output_number>\d+)"
        r"\.npy$"
    )

    @staticmethod
    def parse_output_number_and_filename(
        file_or_number: Union[PathT, int],
        *,
        directory: PathT,
        prefix: str,
    ) -> Tuple[int, Path]:
        directory = Path(directory).resolve()
        if isinstance(file_or_number, (str, Path)):
            file = Path(file_or_number)
            if file == Path(file.name):
                file = directory / "rho" / file

            if (match := NPYReader._filename_re.fullmatch(file.name)) is None:
                raise ValueError(f"Filename {file.name!r} is not recognized")
            output_number = int(match.group("output_number"))
            file_alt = None
        else:
            output_number = file_or_number
            file = directory / "rho" / f"{prefix}_RHO.{output_number:04d}.npy"
            file_alt = file.with_name(f"_{file.name}")

        if not file.is_file():
            if file_alt is not None and file_alt.is_file():
                # backward compatibility
                file = file_alt
            else:
                raise FileNotFoundError(file)
        return output_number, file

    @staticmethod
    def get_bin_files(directory: PathT, /) -> List[Path]:
        # return *all* loadable files
        # (not just the ones matching a particular prefix)
        directory = Path(directory).resolve()
        density_file_paths: List[Path] = []
        for subdir in directory.parent.glob("*"):
            if not subdir.is_dir():
                continue
            if not re.fullmatch(r"\w+", subdir.name):
                continue
            for file in subdir.glob("*.npy"):
                if not file.is_file():
                    continue
                if NPYReader._filename_re.fullmatch(file.name) is None:
                    continue
                density_file_paths.append(file)

        if not density_file_paths:  # pragma: no cover
            raise RuntimeError
        return sorted(density_file_paths)

    @staticmethod
    def read(file: PathT, /, **meta) -> BinData:
        meta.setdefault("prefix", "")

        ref_file = Path(file).resolve()
        if (match := NPYReader._filename_re.fullmatch(ref_file.name)) is None:
            raise ValueError(f"Filename {ref_file.name!r} is not recognized")

        prefix = match.group("prefix")
        output_number = int(match.group("output_number"))

        header = ref_file.parents[1] / "header" / f"header_{prefix}.json"
        with open(header) as fh:
            header_data = json.load(fh)

        geometry = Geometry(header_data.pop("geometry"))
        coordinates: Dict[str, FloatArray] = {
            k: np.array(v, dtype="float32") for k, v in header_data.items()
        }
        x1, x2, x3 = coordinates.values()
        n1, n2, n3 = (len(x) for x in (x1, x2, x3))

        # field discovery
        fields_found: Dict[str, Path] = {}
        haystack = NPYReader.get_bin_files(ref_file.parent)
        for file in haystack:
            match = NPYReader._filename_re.fullmatch(file.name)
            if match is None:  # pragma: no cover
                raise RuntimeError
            if match.group("prefix") != prefix:
                continue
            if int(match.group("output_number")) != output_number:
                continue
            fields_found[match.group("field_name")] = file

        # sanity check: we should have rediscovered our starting file by now
        assert ref_file in fields_found.values()

        data: Dict[str, FloatArray] = {
            k: np.load(v, allow_pickle=True) for k, v in fields_found.items()
        }

        return BinData(data, geometry, x1, x2, x3)
