__all__ = [
    "Fargo3DReader",
    "FargoADSGReader",
    "VTKReader",
]
import json
import os
import re
import sys
from io import BufferedReader
from pathlib import Path
from typing import Any, final

import numpy as np

from nonos._geometry import Geometry
from nonos._types import BinData, F, FArray, FArray1D, FArray3D

if sys.version_info >= (3, 11):
    from typing import assert_never
else:
    from typing_extensions import assert_never
if sys.version_info >= (3, 13):
    from copy import replace
else:
    from dataclasses import replace

f32 = np.float32
f64 = np.float64


def lazy_load_from_fid(
    fid: BufferedReader, /, *, dtype: np.dtype, count: int
) -> FArray1D[F]:
    offset = fid.tell()
    retv = np.memmap(fid, mode="r", dtype=dtype, offset=offset, shape=count)
    fid.seek(offset + dtype.itemsize * count, os.SEEK_SET)
    return retv


@final
class VTKReader:
    NATIVE_COORDINATE_REGEXP = re.compile(r"X(1|2|3)(L|C)_NATIVE_COORDINATES")

    @staticmethod
    def parse_output_number_and_filename(
        file_or_number: os.PathLike[str] | int,
        *,
        directory: os.PathLike[str],
        prefix: str,  # noqa: ARG004
    ) -> tuple[int, Path]:
        if isinstance(file_or_number, int):
            output_number = int(file_or_number)
            file = Path(f"data.{output_number:04d}.vtk")
        else:
            file = Path(file_or_number)
            if (m := re.search(r"\d+", file.name)) is None:
                raise ValueError(
                    f"Failed to parse output number from filename {file.name}"
                )
            else:
                output_number = int(m.group())

        if file == Path(file.name):
            file = Path(directory) / file

        return output_number, file

    @staticmethod
    def get_bin_files(directory: os.PathLike[str], /) -> list[Path]:
        directory = Path(directory)
        return sorted(directory.glob("data.*.vtk"))

    @staticmethod
    def read(file: str | os.PathLike[str], /, **meta: Any) -> BinData[f32]:
        """
        Adapted from Geoffroy Lesur
        Function that reads a vtk file in polar coordinates
        """

        meta.setdefault("geometry", None)
        meta.setdefault("cell", "edges")
        meta.setdefault("computedata", True)

        fid = open(file, "rb")

        # define our datastructure
        f32_be = np.dtype(">f4")  # Big endian single precision floats
        V = BinData.default_init(dtype=f32_be)

        # initialize geometry
        if meta["geometry"] is not None:
            V = replace(V, geometry=Geometry(meta["geometry"]))

        native_coordinates = {}

        dint = np.dtype(">i4")  # Big endian integer

        s = fid.readline()  # VTK DataFile Version x.x
        s = fid.readline()  # Comments

        s = fid.readline()  # BINARY
        s = fid.readline()  # DATASET RECTILINEAR_GRID or STRUCTURED_GRID
        slist = s.split()

        s = fid.readline()  # DIMENSIONS NX NY NZ
        slist = s.split()
        entry = slist[0].decode("utf-8")
        if entry == "FIELD":
            nfield = int(slist[2])
            for _field in range(nfield):
                s = fid.readline()
                slist = s.split()
                entry = slist[0].decode("utf-8")
                if entry == "TIME":
                    # skip
                    fid.seek(f32_be.itemsize, os.SEEK_CUR)
                elif entry == "GEOMETRY":
                    g = np.fromfile(fid, dint, 1)
                    if g == 0:
                        raise NotImplementedError(
                            "Support for cartesian geometry is missing"
                        )
                    elif g == 1:
                        thisgeometry = Geometry.POLAR
                    elif g == 2:
                        thisgeometry = Geometry.SPHERICAL
                    elif g == 3:  # pragma: no cover
                        fid.close()
                        raise NotImplementedError(
                            "Support for cylindrical geometry is missing"
                        )
                    else:  # pragma: no cover
                        fid.close()
                        raise ValueError(
                            f"Unknown value for GEOMETRY flag ({g!r}) was found in the VTK file."
                        )

                    if meta["geometry"] is not None:
                        # We already have a proposed geometry, check that what is read from the file matches
                        if thisgeometry != meta["geometry"]:  # pragma: no cover
                            fid.close()
                            raise ValueError(
                                f"geometry argument ({meta['geometry']!r}) is "
                                "inconsistent with GEOMETRY flag from the VTK file "
                                f"({thisgeometry!r})"
                            )
                    V = replace(V, geometry=thisgeometry)
                elif entry == "PERIODICITY":
                    # skip
                    fid.seek(dint.itemsize * 3, os.SEEK_CUR)
                elif VTKReader.NATIVE_COORDINATE_REGEXP.match(entry):
                    _ncomp, native_dim, _dtype = slist[1:]
                    native_coordinates[entry] = np.fromfile(
                        fid, dtype=f32_be, count=int(native_dim)
                    )
                else:  # pragma: no cover
                    fid.close()
                    raise ValueError(f"Received unknown field: {entry!r}")

                s = fid.readline()  # extra linefeed

            # finished reading the field entry
            # read next line
            s = fid.readline()  # DIMENSIONS...

        if V.geometry is BinData.__dataclass_fields__["geometry"].default:
            fid.close()
            raise RuntimeError(
                "Geometry couldn't be determined from data. "
                "Try to set the geometry keyword argument explicitly."
            )
        slist = s.split()  # DIMENSIONS....
        n1 = int(slist[1])
        n2 = int(slist[2])
        n3 = int(slist[3])

        z: FArray1D[f32]
        r: FArray1D[f32]
        theta: FArray1D[f32]
        phi: FArray1D[f32]

        if V.geometry is Geometry.POLAR or V.geometry is Geometry.SPHERICAL:
            if n3 == 1:
                is2d = 1
            else:
                is2d = 0

            s = fid.readline()  # POINTS NXNYNZ float
            slist = s.split()
            npoints = int(slist[1])
            points = lazy_load_from_fid(fid, dtype=f32_be, count=3 * npoints)
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
            if V.geometry is Geometry.POLAR:
                r = np.sqrt(xcart[:, 0, 0] ** 2 + ycart[:, 0, 0] ** 2)
                theta = np.unwrap(np.arctan2(ycart[0, :, 0], xcart[0, :, 0]))
                z = zcart[0, 0, :]

                s = fid.readline()  # CELL_DATA (NX-1)(NY-1)(NZ-1)
                slist = s.split()
                data_type = slist[0].decode("utf-8")
                if data_type != "CELL_DATA":  # pragma: no cover
                    fid.close()
                    raise ValueError(
                        f"Expected 'CELL DATA' as produced by PLUTO, got {data_type!r}."
                    )
                s = fid.readline()  # Line feed

                if meta["cell"] == "edges":
                    if n1 > 1:
                        n1 -= 1
                    if n2 > 1:
                        n2 -= 1
                    if n3 > 1:
                        n3 -= 1
                    V = replace(V, x1=r, x2=theta, x3=z)

                # Perform averaging on coordinate system to get cell centers
                # The file contains face coordinates, so we extrapolate to get the cell center coordinates.
                elif meta["cell"] == "centers":
                    if n1 > 1:
                        n1 -= 1
                        V = replace(V, x1=0.5 * (r[1:] + r[:-1]))
                    else:
                        V = replace(V, x1=r)
                    if n2 > 1:
                        n2 -= 1
                        V = replace(
                            V,
                            x2=(0.5 * (theta[1:] + theta[:-1]) + np.pi) % (2.0 * np.pi)
                            - np.pi,
                        )
                    else:
                        V = replace(V, x2=theta)
                    if n3 > 1:
                        n3 -= 1
                        V = replace(V, x3=0.5 * (z[1:] + z[:-1]))
                    else:
                        V = replace(V, x3=z)

            # Reconstruct the spherical coordinate system
            elif V.geometry is Geometry.SPHERICAL:
                if is2d:
                    r = np.sqrt(xcart[:, 0, 0] ** 2 + zcart[:, 0, 0] ** 2)
                    phi = np.unwrap(
                        np.arctan2(ycart[0, n2 // 2, :], xcart[0, n2 // 2, :])
                    )
                    Rcyl = np.sqrt(xcart[0, :, 0] ** 2 + zcart[0, :, 0] ** 2)
                    # regularize the denominator to avoid spurious divide-by-zero-warnings
                    Rcyl_reg = np.where(zcart[0, :, 0] == 0, 1.0, Rcyl)
                    theta = np.arccos(zcart[0, :, 0] / Rcyl_reg)
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
                data_type = slist[0].decode("utf-8")
                if data_type != "CELL_DATA":  # pragma: no cover
                    fid.close()
                    raise ValueError(
                        f"Expected 'CELL DATA' as produced by PLUTO. Got {data_type!r}"
                    )
                s = fid.readline()  # Line feed

                if meta["cell"] == "edges":
                    if n1 > 1:
                        n1 -= 1
                    if n2 > 1:
                        n2 -= 1
                    if n3 > 1:
                        n3 -= 1
                    V = replace(V, x1=r, x2=theta, x3=phi)

                # Perform averaging on coordinate system to get cell centers
                # The file contains face coordinates, so we extrapolate to get the cell center coordinates.
                elif meta["cell"] == "centers":
                    if n1 > 1:
                        n1 -= 1
                        V = replace(V, x1=0.5 * (r[1:] + r[:-1]))
                    else:
                        V = replace(V, x1=r)
                    if n2 > 1:
                        n2 -= 1
                        V = replace(V, x2=0.5 * (theta[1:] + theta[:-1]))
                    else:
                        V = replace(V, x2=theta)
                    if n3 > 1:
                        n3 -= 1
                        V = replace(V, x3=0.5 * (phi[1:] + phi[:-1]))
                    else:
                        V = replace(V, x3=phi)
            else:
                assert_never(V.geometry)
        else:
            assert_never(V.geometry)

        if meta["computedata"]:
            new_shape = n3, n2, n1
            grid_size = int(np.prod(new_shape))
            while True:
                # SCALARS/VECTORS name data_type (ex: SCALARS imagedata unsigned_char)
                s = fid.readline()
                if len(s) < 2:  # leave if end of file
                    break
                slist = s.split()
                datatype = slist[0].decode("utf-8")
                varname = slist[1].decode("utf-8").upper()
                if datatype == "SCALARS":
                    fid.readline()  # LOOKUP TABLE
                    array = lazy_load_from_fid(
                        fid,
                        dtype=f32_be,
                        count=grid_size,
                    ).reshape(new_shape)
                    V.data[varname] = np.transpose(array)
                elif datatype == "VECTORS":
                    Q = lazy_load_from_fid(fid, dtype=f32_be, count=3 * grid_size)

                    for i, suffix in enumerate("XYZ"):
                        name = f"{varname}_{suffix}"
                        V.data[name] = np.transpose(Q[i::3].reshape(new_shape))

                else:  # pragma: no cover
                    fid.close()
                    raise ValueError(
                        f"Unknown datatype {datatype!r}. "
                        "Expected 'SCALARS' or 'VECTORS'"
                    )

                fid.readline()  # extra line feed
        fid.close()

        if meta["cell"] == "edges":
            native2attr = {
                "X1L_NATIVE_COORDINATES": "x1",
                "X2L_NATIVE_COORDINATES": "x2",
                "X3L_NATIVE_COORDINATES": "x3",
            }
        elif meta["cell"] == "centers":
            native2attr = {
                "X1C_NATIVE_COORDINATES": "x1",
                "X2C_NATIVE_COORDINATES": "x2",
                "X3C_NATIVE_COORDINATES": "x3",
            }
        else:  # pragma: no cover
            raise ValueError(
                f"Unknown cell type {meta['cell']!r}. Expected 'edges' or 'centers'"
            )

        if native_coordinates:
            for native_field, attr in native2attr.items():
                V = replace(V, **{attr: native_coordinates[native_field]})  # type: ignore

        return V.finalize()


@final
class FargoReaderHelper:
    @staticmethod
    def parse_output_number_and_filename(
        file_or_number: os.PathLike[str] | int,
        *,
        directory: os.PathLike[str],
        prefix: str,  # noqa ARG004
    ) -> tuple[int, Path]:
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
    def get_bin_files(directory: os.PathLike[str], /) -> list[Path]:
        directory = Path(directory)
        return [
            fn
            for fn in sorted(directory.glob("gasdens*.dat"))
            if re.search(r"gasdens\d+.dat$", str(fn)) is not None
        ]

    @staticmethod
    def _get_output_number_and_dir_from(
        file: str | os.PathLike[str],
    ) -> tuple[int, Path]:
        _in_file = Path(file).resolve()
        directory = _in_file.parent
        if (match := re.search(r"(?P<on>\d+).dat$", _in_file.name)) is not None:
            output_number = int(match.group("on"))
        else:
            raise ValueError(f"Failed to parse filename {file!r}")

        return output_number, directory


@final
class Fargo3DReader:
    @staticmethod
    def parse_output_number_and_filename(
        file_or_number: os.PathLike[str] | int,
        *,
        directory: os.PathLike[str],
        prefix: str,
    ) -> tuple[int, Path]:
        return FargoReaderHelper.parse_output_number_and_filename(
            file_or_number, directory=directory, prefix=prefix
        )

    @staticmethod
    def get_bin_files(directory: os.PathLike[str], /) -> list[Path]:
        return FargoReaderHelper.get_bin_files(directory)

    @staticmethod
    def read(
        file: os.PathLike[str],
        /,
        **meta: Any,
    ) -> BinData[f64]:
        output_number, directory = FargoReaderHelper._get_output_number_and_dir_from(
            file
        )

        default_fluid = "gas"
        fluid_option: str | None = meta.get("fluid", default_fluid)
        if fluid_option is None:
            fluid = default_fluid
        else:
            fluid = fluid_option

        V = BinData.default_init(dtype=np.dtype("float64"))
        geometry_str = meta["COORDINATES"]

        domain_x = np.loadtxt(directory / "domain_x.dat", dtype="float64")
        # We avoid ghost cells
        domain_y = np.loadtxt(directory / "domain_y.dat", dtype="float64")[3:-3]
        domain_z = np.loadtxt(directory / "domain_z.dat", dtype="float64")
        if domain_z.shape[0] > 6:
            domain_z = domain_z[3:-3]

        V = replace(V, x1=domain_y)  # X-Edge
        if geometry_str == "cylindrical":
            V = replace(
                V,
                geometry=Geometry.POLAR,
                x2=domain_x,  # Y-Edge
                x3=domain_z,  # Z-Edge #latitude
            )
            pairs = [("RHO", "dens"), ("VX1", "vy"), ("VX2", "vx"), ("VX3", "vz")]
        elif geometry_str == "spherical":
            V = replace(
                V,
                geometry=Geometry.SPHERICAL,
                x2=domain_z,  # Z-Edge #latitude
                x3=domain_x,  # Y-Edge
            )
            pairs = [("RHO", "dens"), ("VX1", "vy"), ("VX2", "vz"), ("VX3", "vx")]
        else:
            raise NotImplementedError(f"Geometry {geometry_str!r} is not supported")

        n1 = len(V.x1) - 1
        n2 = len(V.x2) - 1
        n3 = len(V.x3) - 1
        if geometry_str == "cylindrical":
            grid_shape = n3, n1, n2
            tuple_transposition = (1, 2, 0)
            shift = n2 // 2
        elif geometry_str == "spherical":
            grid_shape = n2, n1, n3
            tuple_transposition = (1, 0, 2)
            shift = n3 // 2
        else:
            raise NotImplementedError(f"Geometry {geometry_str!r} is not supported")

        def _read_array(file: Path) -> FArray3D[F]:
            return np.roll(
                np.fromfile(file, dtype="float64")
                .reshape(grid_shape)
                .transpose(tuple_transposition),
                shift,
                axis=1,
            )

        for key, field in pairs:
            file = directory / f"{fluid}{field}{output_number}.dat"
            if not file.is_file():
                continue
            V.data[key] = _read_array(file)

        if not V.data:
            raise FileNotFoundError(
                f"No file matches the pattern '{fluid}*{output_number}.dat'"
            )

        return V.finalize()


@final
class FargoADSGReader:
    @staticmethod
    def parse_output_number_and_filename(
        file_or_number: os.PathLike[str] | int,
        *,
        directory: os.PathLike[str],
        prefix: str,
    ) -> tuple[int, Path]:
        return FargoReaderHelper.parse_output_number_and_filename(
            file_or_number, directory=directory, prefix=prefix
        )

    @staticmethod
    def get_bin_files(directory: os.PathLike[str], /) -> list[Path]:
        return FargoReaderHelper.get_bin_files(directory)

    @staticmethod
    def read(
        file: os.PathLike[str],
        /,
        **meta: Any,  # noqa: ARG004
    ) -> BinData[f64]:
        output_number, directory = FargoReaderHelper._get_output_number_and_dir_from(
            file
        )

        V = BinData.default_init(dtype=np.dtype("float64"))
        V = replace(V, geometry=Geometry.POLAR)

        phi = np.loadtxt(directory / "used_azi.dat", dtype="float64")[:, 0]
        domain_x = np.zeros(len(phi) + 1, dtype="float64")
        domain_x[:-1] = phi
        domain_x[-1] = 2 * np.pi
        domain_x -= np.pi
        # We avoid ghost cells
        domain_y = np.loadtxt(directory / "used_rad.dat", dtype="float64")
        domain_z = np.zeros(2, dtype="float64")

        V = replace(
            V,
            x1=domain_y,  # X-Edge
            x2=domain_x,  # Y-Edge
            x3=domain_z,  # Z-Edge #latitude
        )

        n1 = len(V.x1) - 1
        n2 = len(V.x2) - 1
        n3 = len(V.x3) - 1
        grid_shape = n3, n1, n2

        def _read_array(file: Path) -> FArray3D[F]:
            return (  # type: ignore[no-any-return]
                np.fromfile(file, dtype="float64")
                .reshape(grid_shape)
                .transpose(1, 2, 0)
            )

        for key, field in [("RHO", "dens"), ("VX1", "vrad"), ("VX2", "vtheta")]:
            file = directory / f"gas{field}{output_number}.dat"
            if not file.is_file():
                continue
            V.data[key] = _read_array(file)
        return V.finalize()


class NPYReader:
    # we accept a leading '_' for backward compatibility
    _filename_re = re.compile(
        r"^_?(?P<prefix>[\w\.]*)"
        r"_(?P<field_name>[A-Z\d]+)"
        r"\.(?P<output_number>\d+)"
        r"\.npy$"
    )

    @staticmethod
    def parse_output_number_and_filename(
        file_or_number: os.PathLike[str] | int,
        *,
        directory: os.PathLike[str],
        prefix: str,
    ) -> tuple[int, Path]:
        directory = Path(directory).resolve()
        if isinstance(file_or_number, int):
            output_number = file_or_number
            all_bin_files = NPYReader.get_bin_files(directory / "any")
            _filter_re = re.compile(rf"^_?{prefix}_[A-Z\d]+.{output_number:04d}.npy")
            matches = [
                file for file in all_bin_files if _filter_re.fullmatch(file.name)
            ]
            if not matches:
                raise FileNotFoundError(
                    f"Failed to locate a file matching {prefix=!r} and {output_number=}"
                )
            file = matches[0]
            file_alt = (
                None if file.name.startswith("_") else file.with_name(f"_{file.name}")
            )
        else:
            file = Path(file_or_number)
            if (match := NPYReader._filename_re.fullmatch(file.name)) is None:
                raise ValueError(f"Filename {file.name!r} is not recognized")
            if file == Path(file.name):
                file = directory / match.group("field_name").lower() / file
            output_number = int(match.group("output_number"))
            file_alt = None

        if not file.is_file():
            if file_alt is None:
                raise FileNotFoundError(str(file))

            if not file_alt.is_file():
                raise FileNotFoundError(f"{file} (also tried {file_alt})")
            # backward compatibility
            file = file_alt

        return output_number, file

    @staticmethod
    def get_bin_files(directory: os.PathLike[str], /) -> list[Path]:
        # return *all* loadable files
        # (not just the ones matching a particular prefix)
        directory = Path(directory).resolve()
        file_paths: list[Path] = []
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
                file_paths.append(file)

        return sorted(file_paths)

    @staticmethod
    def read(file: os.PathLike[str], /, **meta: Any) -> BinData[f32]:
        meta.setdefault("prefix", "")

        ref_file = Path(file).resolve()
        if (match := NPYReader._filename_re.fullmatch(ref_file.name)) is None:
            raise ValueError(f"Filename {ref_file.name!r} is not recognized")

        prefix = match.group("prefix")
        output_number = int(match.group("output_number"))

        op_suffix = f"_{prefix}" if prefix else ""
        header = ref_file.parents[1] / "header" / f"header{op_suffix}.json"
        with open(header) as fh:
            header_data = json.load(fh)

        geometry = Geometry(header_data.pop("geometry"))
        coordinates: dict[str, FArray1D[f32]] = {
            k: np.array(v, dtype="float32") for k, v in header_data.items()
        }
        x1, x2, x3 = coordinates.values()
        n1, n2, n3 = (len(x) for x in (x1, x2, x3))

        # field discovery
        fields_found: dict[str, Path] = {}
        haystack = NPYReader.get_bin_files(ref_file.parent)
        for file in haystack:
            match = NPYReader._filename_re.fullmatch(file.name)
            if match is None:
                raise AssertionError
            if match.group("prefix") != prefix:
                continue
            if int(match.group("output_number")) != output_number:
                continue
            fields_found[match.group("field_name")] = file

        # sanity check: we should have rediscovered our starting file by now
        assert ref_file in fields_found.values()

        data: dict[str, FArray[Any, f32]] = {
            k: np.load(v, allow_pickle=True) for k, v in fields_found.items()
        }

        return BinData(data, geometry, x1, x2, x3, dtype=np.dtype("float32"))
