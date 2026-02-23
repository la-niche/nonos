import dataclasses
import json
import os
import sys
import warnings
from collections import deque
from collections.abc import ItemsView, KeysView, ValuesView
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from shutil import copyfile
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    cast,
    final,
    overload,
)

import numpy as np
from matplotlib.scale import SymmetricalLogTransform
from matplotlib.ticker import SymmetricalLogLocator

from nonos._geometry import (
    Axis,
    Coordinates,
    Geometry,
    axes_from_geometry,
)
from nonos._integrity_checks import (
    collect_dtype_exceptions,
    collect_shape_exceptions,
    compile_exceptions,
)
from nonos._readers.binary import NPYReader
from nonos._types import (
    D1,
    D2,
    D,
    F,
    FArray,
    FArray0D,
    FArray1D,
    FArray2D,
    FArray3D,
    PlanetData,
)
from nonos.api._angle_parsing import (
    _resolve_planet_file,
    _resolve_rotate_by,
)
from nonos.api.tools import bracketing_values, closest_index, closest_value
from nonos.loaders import BUILTIN_RECIPES, Loader

if sys.version_info >= (3, 11):
    from typing import TypedDict, Unpack, assert_never
else:
    from typing_extensions import TypedDict, Unpack, assert_never

if sys.version_info >= (3, 13):
    from warnings import deprecated
else:
    from typing_extensions import deprecated

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure


@dataclass(frozen=True, eq=False, slots=True)
class NamedArray(Generic[D, F]):
    name: str
    data: FArray[D, F]


class PColorMeshKwargs(TypedDict, total=False):
    vmin: float | None
    vmax: float | None
    norm: "Normalize"
    alpha: float
    shading: Literal["flat", "nearest", "gouraud", "auto"] | None
    rasterized: bool


class Plotable(Generic[D, F]):
    __slots__ = ["abscissa", "ordinate", "field"]

    def __init__(
        self,
        *,
        abscissa: tuple[str, FArray[D, F]],
        ordinate: tuple[str, FArray[D, F]],
        field: tuple[str, FArray[D, F]] | None = None,
    ) -> None:
        self.abscissa: NamedArray[D, F] = NamedArray(*abscissa)
        self.ordinate: NamedArray[D, F] = NamedArray(*ordinate)
        self.field: NamedArray[D, F] | None = (
            None if field is None else NamedArray(*field)
        )
        if ndim := self.data.ndim > 2:
            raise TypeError(
                f"Plotable doesn't support data with dimensionality>2, got {ndim}"
            )

    @property
    def data(self) -> FArray[D, F]:
        if self.field is not None:
            arr = self.field.data
            assert arr.ndim == 2
        else:
            arr = self.ordinate.data
            assert arr.ndim == 1

        return arr

    def plot(
        self,
        fig: "Figure",
        ax: "Axes",
        *,
        log: bool = False,
        cmap: str | None = "inferno",
        filename: str | None = None,
        fmt: str = "png",
        dpi: int = 500,
        title: str | None = None,
        unit_conversion: float | None = None,
        nbin: int | None = None,  # deprecated
        **kwargs: Unpack[PColorMeshKwargs],
    ) -> "Artist":
        if nbin is not None:
            warnings.warn(
                "The nbin parameter has no effect and is deprecated",
                stacklevel=2,
            )
        data = self.data
        if unit_conversion is not None:
            data = data * unit_conversion
        if log:
            data = np.log10(data)

        akey = self.abscissa.name
        aval = self.abscissa.data
        okey = self.ordinate.name
        oval = self.ordinate.data

        artist: Artist
        if data.ndim == 2:
            kw: PColorMeshKwargs = {}
            if (norm := kwargs.get("norm")) is not None:
                if "vmin" in kwargs:
                    norm.vmin = kwargs.pop("vmin")
                if "vmax" in kwargs:
                    norm.vmax = kwargs.pop("vmax")
            else:
                vmin = (
                    kwargs.pop("vmin") if "vmin" in kwargs else float(np.nanmin(data))
                )
                vmax = (
                    kwargs.pop("vmax") if "vmax" in kwargs else float(np.nanmax(data))
                )
                kw.update({"vmin": vmin, "vmax": vmax})

            artist = im = ax.pcolormesh(aval, oval, data, cmap=cmap, **(kwargs | kw))  # ty: ignore[invalid-argument-type]
            ax.set(
                xlim=(aval.min(), aval.max()),
                ylim=(oval.min(), oval.max()),
                xlabel=akey,
                ylabel=okey,
            )
            if title is not None:
                from mpl_toolkits.axes_grid1 import (  # type: ignore[import-untyped]
                    make_axes_locatable,
                )

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(
                    im, cax=cax, orientation="vertical"
                )  # , format='%.0e')
                cbar.set_label(title)

                cb_axis = cbar.ax.yaxis
                if cb_axis.get_scale() == "symlog":
                    # no minor tick is drawn in symlog norms by default
                    # as of matplotlib 3.7.1, see
                    # https://github.com/matplotlib/matplotlib/issues/25994
                    trf = cb_axis.get_transform()
                    if not isinstance(trf, SymmetricalLogTransform):
                        raise AssertionError
                    cb_axis.set_major_locator(SymmetricalLogLocator(trf))
                    if float(trf.base).is_integer():
                        locator = SymmetricalLogLocator(
                            trf, subs=list(range(1, int(trf.base)))
                        )
                        cb_axis.set_minor_locator(locator)
        elif data.ndim == 1:
            vmin = kwargs.pop("vmin") if "vmin" in kwargs else float(np.nanmin(data))
            vmax = kwargs.pop("vmax") if "vmax" in kwargs else float(np.nanmax(data))
            if "norm" in kwargs:
                kwargs.pop("norm")
            artist = ax.plot(aval, data, **kwargs)[0]
            ax.set(ylim=(vmin, vmax), xlabel=akey)
            if title is not None:
                ax.set_ylabel(title)
        else:
            raise TypeError(
                f"Plotable doesn't support data with dimensionality>2, got {data.ndim}"
            )
        if filename is not None:
            fig.savefig(f"{filename}.{fmt}", bbox_inches="tight", dpi=dpi)

        return artist


def _get_ind_snapshot_uid(
    loader: Loader[F], snapshot_uid: int, time: FArray1D[F]
) -> int:
    ini = loader.load_ini_file()
    target_time = ini.output_time_interval * snapshot_uid
    return closest_index(time, target_time)


def _find_planet_azimuth(
    loader: Loader[F],
    snapshot_uid: int,
    *,
    planet_file: str,
) -> float:
    data_dir = loader.parameter_file.parent
    pd = loader.load_planet_data(data_dir / planet_file)
    ind_on = _get_ind_snapshot_uid(loader, snapshot_uid, pd.t)
    return float(np.arctan2(pd.y, pd.x)[ind_on] % (2 * np.pi))


class FieldAttrs(Generic[F], TypedDict, total=False):
    name: str
    data: FArray3D[F]
    coordinates: Coordinates[F]
    native_geometry: Geometry
    snapshot_uid: int
    loader: Loader[F]
    operation: str


@final
@dataclass(slots=True, frozen=True, kw_only=True)
class Field(Generic[F]):
    name: str
    data: FArray3D[F]
    coordinates: Coordinates[F]
    native_geometry: Geometry
    snapshot_uid: int
    loader: Loader[F]
    operation: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", self.data.view())
        self.data.flags.writeable = False

        if excs := compile_exceptions(
            "multiple issues with inputs",
            (
                "multiple issues with input shapes",
                collect_shape_exceptions(
                    self.shape,
                    self.coordinates.shape,
                ),
            ),
            (
                "multiple issues with input dtypes",
                collect_dtype_exceptions(
                    self.data.dtype,
                    self.coordinates.dtype,
                ),
            ),
        ):
            raise excs

    @property
    def snapshot_number(self) -> int:  # pragma: no cover
        return self.snapshot_uid

    @property
    @deprecated(
        "(Gas)Field.field is deprecated since v0.20.0, "
        "and may be removed in a future version. "
        "Use Field.name instead"
    )
    def field(self) -> str:  # pragma: no cover
        return self.name

    @property
    @deprecated(
        "Field.inifile  is deprecated since v0.20.0, "
        "and may be removed in a future version. "
        "Use Field.loader.parameter_file instead"
    )
    def inifile(self) -> Path:  # pragma: no cover
        return self.loader.parameter_file

    @property
    @deprecated(
        "(Gas)Field.coords is deprecated since v0.20.0, "
        "and may be removed in a future version. "
        "Use Field.coordinates instead"
    )
    def coords(self) -> Coordinates[F]:
        return self.coordinates

    @property
    @deprecated(
        "(Gas)Field.on is deprecated since v0.20.0, "
        "and may be removed in a future version. "
        "Use Field.snapshot_uid instead"
    )
    def on(self) -> int:  # pragma: no cover
        return self.snapshot_uid

    @property
    @deprecated(
        "(Gas)Field.directory is deprecated since v0.20.0, "
        "and may be removed in a future version. "
        "Use Field.loader.parameter_file.parent instead"
    )
    def directory(self) -> Path:  # pragma: no cover
        return self.loader.parameter_file.parent

    def replace(self, **substitutions: Unpack[FieldAttrs[F]]) -> "Field[F]":
        """Convenience wrapper around copy.replace"""
        if sys.version_info >= (3, 13):
            from copy import replace
        else:
            from dataclasses import replace
        return replace(self, **substitutions)

    @property
    def dtype(self) -> np.dtype[F]:
        return self.loader.components.dtype

    @property
    def shape(self) -> tuple[int, int, int]:
        i, j, k = (max(1, n - 1) for n in self.coordinates.shape)
        return i, j, k

    @property
    def _reduced_dimensions(self) -> tuple[bool, bool, bool]:
        # 1 indicates a dimension that is effectively reduced
        s = self.shape
        return s[0] == 1, s[1] == 1, s[2] == 1

    @property
    def effective_ndim(self) -> Literal[0, 1, 2, 3]:
        """
        The effective dimensionality of the underlying data.
        This corresponds to the number of dimensions with more than a single element.

        .. versionadded: 0.20.0
        """
        one_count = sum(self._reduced_dimensions)
        assert 0 <= one_count <= 3
        return cast(Literal[1, 2, 3], 3 - one_count)

    @overload
    def as_ndview(self, *, ndim: Literal[0]) -> FArray0D[F]: ...
    @overload
    def as_ndview(self, *, ndim: Literal[1]) -> FArray1D[F]: ...
    @overload
    def as_ndview(self, *, ndim: Literal[2]) -> FArray2D[F]: ...
    @overload
    def as_ndview(self, *, ndim: Literal[3]) -> FArray3D[F]: ...
    def as_ndview(self, *, ndim):  # type: ignore[no-untyped-def]
        """
        Create a read-only view of the underlying data array with a specific number of dimensions.

        Parameters
        ----------
        ndim: 0, 1, 2, or 3 (keyword-only)
          The desired number of dimensions of the result

        Raises
        ------
        TypeError: if the effective number of dimensions of the data is greater than ndim
        ValueError: if ndim is anything other than 1, 2 or 3

        .. versionadded: 0.20.0
        """
        if ndim not in {0, 1, 2, 3}:
            raise ValueError(f"Expected ndim to be either 0, 1, 2 or 3. Got {ndim=}")
        if (eff_ndim := self.effective_ndim) > ndim:
            raise TypeError(f"Effective ndim {eff_ndim} is greater than target {ndim}")
        if eff_ndim < ndim < 3:
            # not clear how this *should* work: we'd be left with one or more dangling
            # dimensions whose positions could be ambiguous.
            # Better to error out if this isn't needed.
            raise TypeError(
                f"Cannot produce a {ndim}d view from effective ndim {eff_ndim}"
            )
        match ndim:
            case 3:
                arr = self.data.view()
            case _:
                arr = self.data.squeeze()

        if arr.ndim != ndim:
            raise AssertionError

        return arr

    def as_0dview(self) -> FArray0D[F]:
        """
        Shorthand for as_ndview(ndim=0)

        .. versionadded: 0.20.0
        """
        return self.as_ndview(ndim=0)

    def as_1dview(self) -> FArray1D[F]:
        """
        Shorthand for as_ndview(ndim=1)

        .. versionadded: 0.20.0
        """
        return self.as_ndview(ndim=1)

    def as_2dview(self) -> FArray2D[F]:
        """
        Shorthand for as_ndview(ndim=2)

        .. versionadded: 0.20.0
        """
        return self.as_ndview(ndim=2)

    def as_3dview(self) -> FArray3D[F]:
        """
        Shorthand for as_ndview(ndim=3)

        .. versionadded: 0.20.0
        """
        return self.as_ndview(ndim=3)

    @overload
    def map(
        self,
        a: str,
        b: None,
        /,
        rotate_by: float | None = None,
        rotate_with: str | None = None,
    ) -> Plotable[D1, F]: ...
    @overload
    def map(
        self,
        a: str,
        b: str,
        /,
        rotate_by: float | None = None,
        rotate_with: str | None = None,
    ) -> Plotable[D2, F]: ...
    def map(  # type: ignore[no-untyped-def]
        self,
        a,
        b=None,
        /,
        rotate_by=None,  # deprecated
        rotate_with=None,  # deprecated
    ):
        data_key = self.name
        if self.effective_ndim > 2:
            raise ValueError("data has to be 1D or 2D in order to call map.")

        axis_1 = Axis.from_label(a)
        if rotate_by is not None:
            warnings.warn(
                "the rotate_by argument is deprecated. since v0.20.0 "
                "and might be removed in a future version. "
                "Use Field.azimuthal_rotation instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if rotate_with is not None:
            warnings.warn(
                "the rotate_with argument is deprecated since v0.20.0 "
                "and might be removed in a future version. "
                "Use Field.azimuthal_rotation instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        if b is None:
            meshgrid_conversion = self.coordinates._meshgrid_conversion_1d(axis_1)

            abscissa_value = list(meshgrid_conversion.values())[0]
            abscissa_key = list(meshgrid_conversion.keys())[0]
            if axis_1 is Axis.AZIMUTH:
                data = self.azimuthal_rotation(
                    rotate_by=rotate_by,
                    rotate_with=rotate_with,
                ).data
            else:
                data = self.data.view()

            return Plotable(
                abscissa=(abscissa_key.label, abscissa_value),
                ordinate=(data_key, data.squeeze()),
            )

        else:
            axis_2 = Axis.from_label(b)

            meshgrid_conversion = self.coordinates._meshgrid_conversion_2d(
                axis_1, axis_2
            )
            # meshgrid in polar coordinates P, R (if "R", "phi") or R, P (if "phi", "R")
            # idem for all combinations of R,phi,z
            abscissa_value, ordinate_value = (
                meshgrid_conversion[axis_1],
                meshgrid_conversion[axis_2],
            )
            abscissa_key, ordinate_key = (axis_1, axis_2)
            native_plane_axes = self.coordinates.native_from_wanted(axis_1, axis_2)
            if Axis.AZIMUTH in native_plane_axes:
                data = self.azimuthal_rotation(
                    rotate_by=rotate_by,
                    rotate_with=rotate_with,
                ).data
            else:
                data = self.data.view()

            def rotate_axes(arr: FArray[D, F], shift: int) -> FArray[D, F]:
                axes_in = tuple(range(arr.ndim))
                axes_out = deque(axes_in)
                axes_out.rotate(shift)
                return np.moveaxis(arr, axes_in, axes_out)

            # make reduction axis the first axis then drop (squeeze) it,
            # while preserving the original (cyclic) order in the other two axes
            data = rotate_axes(data, shift=self.shape.index(1)).squeeze()

            naxes = axes_from_geometry(self.native_geometry)
            sorted_pairs = [
                (naxes[0], naxes[1]),
                (naxes[1], naxes[2]),
                (naxes[2], naxes[0]),
            ]
            if native_plane_axes in sorted_pairs:
                data = data.T

            return Plotable(
                abscissa=(abscissa_key.label, abscissa_value),
                ordinate=(ordinate_key.label, ordinate_value),
                field=(data_key, data),
            )

    def save(
        self,
        directory: os.PathLike[str] | None = None,
        header_only: bool = False,
    ) -> Path:
        if directory is None:
            directory = Path.cwd()
        else:
            directory = Path(directory)
        operation = self.operation
        header_dir = directory / "header"
        subdir = directory / self.name.lower()
        full_name = f"{operation}{'_' if operation else ''}{self.name}"
        file = subdir / f"{full_name}.{self.snapshot_uid:04d}.npy"
        if not (header_only or file.is_file()):
            subdir.mkdir(exist_ok=True, parents=True)
            np.save(file, self.data)

        group_of_files = list(subdir.glob(f"{operation}*"))
        op_suffix = f"{'_' if operation else ''}{operation}"
        filename = f"header{op_suffix}.json"
        header_file = header_dir / filename
        if (len(group_of_files) > 0 and not header_file.is_file()) or header_only:
            header_dir.mkdir(exist_ok=True, parents=True)
            if not header_file.is_file():
                dictsaved = self.coordinates.to_dict()

                def is_array(item: tuple[str, Any]) -> bool:
                    _key, value = item
                    return isinstance(value, np.ndarray)

                for key, value in filter(is_array, dictsaved.items()):
                    dictsaved[key] = value.tolist()
                with open(header_file, "w") as hfile:
                    json.dump(dictsaved, hfile, indent=2)

        src = self.loader.parameter_file
        dest = directory / self.loader.parameter_file.name
        if dest != src:
            copyfile(src, dest)

        return file

    def find_ir(self, distance: float = 1.0) -> int:
        match self.native_geometry:
            case Geometry.POLAR:
                return closest_index(
                    self.coordinates.get_axis_array_med(Axis.CYLINDRICAL_RADIUS),
                    distance,
                )
            case Geometry.SPHERICAL:
                return closest_index(
                    self.coordinates.get_axis_array_med(Axis.SPHERICAL_RADIUS),
                    distance,
                )
            case Geometry.CARTESIAN:
                raise NotImplementedError
            case _ as unreachable:
                assert_never(unreachable)

    def find_imid(self, altitude: float = 0.0) -> int:
        match self.native_geometry:
            case Geometry.CARTESIAN | Geometry.POLAR:
                arr = self.coordinates.get_axis_array_med(Axis.CARTESIAN_Z)
                return closest_index(arr, altitude)
            case Geometry.SPHERICAL:
                arr = self.coordinates.get_axis_array_med(Axis.COLATITUDE)
                return closest_index(arr, np.pi / 2 - altitude)
            case _ as unreachable:
                assert_never(unreachable)

    def find_iphi(self, phi: float = 0) -> int:
        match self.native_geometry:
            case Geometry.POLAR | Geometry.SPHERICAL:
                phiarr = self.coordinates.get_axis_array(Axis.AZIMUTH)
                mod = len(phiarr) - 1
                return closest_index(phiarr, phi) % mod
            case Geometry.CARTESIAN:
                raise NotImplementedError
            case _ as unreachable:
                assert_never(unreachable)

    def _load_planet(
        self,
        *,
        planet_number: int | None = None,
        planet_file: str | None = None,
    ) -> PlanetData[F]:
        planet_file = _resolve_planet_file(
            planet_number=planet_number, planet_file=planet_file
        )
        file = self.loader.parameter_file.parent / planet_file
        return self.loader.load_planet_data(file)

    def _get_ind_snapshot_uid(self, time: FArray1D[F]) -> int:
        return _get_ind_snapshot_uid(self.loader, self.snapshot_uid, time)

    def find_rp(
        self,
        planet_number: int | None = None,
        *,
        planet_file: str | None = None,
    ) -> float:
        pd = self._load_planet(planet_number=planet_number, planet_file=planet_file)
        ind_on = self._get_ind_snapshot_uid(pd.t)
        return float(pd.d[ind_on])  # type: ignore [attr-defined]

    def find_rhill(
        self,
        planet_number: int | None = None,
        *,
        planet_file: str | None = None,
    ) -> float:
        ini = self.loader.load_ini_file()
        pd = self._load_planet(planet_number=planet_number, planet_file=planet_file)
        oe = pd.get_orbital_elements(ini.frame)
        ind_on = self._get_ind_snapshot_uid(pd.t)
        return float(pd.q[ind_on] / 3.0 ** (1 / 3) * oe.a[ind_on])

    def find_phip(
        self,
        planet_number: int | None = None,
        *,
        planet_file: str | None = None,
    ) -> float:
        return _find_planet_azimuth(
            self.loader,
            self.snapshot_uid,
            planet_file=_resolve_planet_file(
                planet_file=planet_file,
                planet_number=planet_number,
            ),
        )

    @staticmethod
    def _resolve_operation_name(
        *,
        prefix: str,
        default_suffix: str,
        operation_name: str | None,
    ) -> str:
        if operation_name == "":
            raise ValueError("operation_name cannot be empty")
        suffix = operation_name or default_suffix
        if prefix:
            return f"{prefix}_{suffix}"
        else:
            return suffix

    def latitudinal_projection(
        self,
        theta: float | None = None,
        *,
        operation_name: str | None = None,
    ) -> "Field[F]":
        default_suffix = "latitudinal_projection"
        if theta is not None:
            default_suffix += str(np.pi / 2 - theta)
        operation = self._resolve_operation_name(
            prefix=self.operation,
            default_suffix=default_suffix,
            operation_name=operation_name,
        )

        imid = self.find_imid()
        match self.native_geometry:
            case Geometry.CARTESIAN:
                raise NotImplementedError(
                    "latitudinal_projection isn't implemented for cartesian geometry"
                )
            case Geometry.POLAR:
                ret_coords = Coordinates(
                    self.native_geometry,
                    self.coordinates.get_axis_array("R"),
                    self.coordinates.get_axis_array("phi"),
                    bracketing_values(
                        self.coordinates.get_axis_array("z"),
                        self.coordinates.get_axis_array_med("z")[imid],
                    ).as_array(dtype=self.dtype),
                )
                R = self.coordinates.get_axis_array_med("R")
                z = self.coordinates.get_axis_array_med("z")
                integral = np.zeros((self.shape[0], self.shape[1]), dtype=">f4")
                for i in range(self.shape[0]):
                    km = closest_index(z, z.min())
                    kp = closest_index(z, z.max())
                    if theta is not None:
                        km = closest_index(z, -R[i] * theta)
                        kp = closest_index(z, R[i] * theta)
                    integral[i, :] = (
                        self.data[i, :, :]
                        * np.ediff1d(self.coordinates.get_axis_array("z"))[None, :]
                    )[:, km : kp + 1].sum(axis=1, dtype="float64")
                ret_data = integral.reshape(self.shape[0], self.shape[1], 1)
            case Geometry.SPHERICAL:
                ret_coords = Coordinates(
                    self.native_geometry,
                    self.coordinates.get_axis_array("r"),
                    bracketing_values(
                        self.coordinates.get_axis_array("theta"),
                        self.coordinates.get_axis_array_med("theta")[imid],
                    ).as_array(dtype=self.dtype),
                    self.coordinates.get_axis_array("phi"),
                )
                km = closest_index(
                    self.coordinates.get_axis_array("theta"),
                    self.coordinates.get_axis_array("theta").min(),
                )
                kp = closest_index(
                    self.coordinates.get_axis_array("theta"),
                    self.coordinates.get_axis_array("theta").max(),
                )
                if theta is not None:
                    kp = closest_index(
                        self.coordinates.get_axis_array("theta"), np.pi / 2 + theta
                    )
                    km = closest_index(
                        self.coordinates.get_axis_array("theta"), np.pi / 2 - theta
                    )
                ret_data = (
                    (
                        self.data
                        * self.coordinates.get_axis_array_med("r")[:, None, None]
                        * np.sin(
                            self.coordinates.get_axis_array_med("theta")[None, :, None]
                        )
                        * np.ediff1d(self.coordinates.get_axis_array("theta"))[
                            None, :, None
                        ]
                    )[:, km : kp + 1, :]
                    .sum(axis=1, dtype="float64")
                    .reshape(self.shape[0], 1, self.shape[2])
                )
            case _ as unreachable:
                assert_never(unreachable)

        return self.replace(
            data=ret_data.astype(self.dtype, copy=False),
            coordinates=ret_coords,
            operation=operation,
        )

    def vertical_projection(
        self,
        z: float | None = None,
        *,
        operation_name: str | None = None,
    ) -> "Field[F]":
        default_suffix = "vertical_projection"
        if z is not None:
            default_suffix += str(z)
        operation = self._resolve_operation_name(
            prefix=self.operation,
            default_suffix=default_suffix,
            operation_name=operation_name,
        )

        imid = self.find_imid()
        match self.native_geometry:
            case Geometry.CARTESIAN:
                zarr = self.coordinates.get_axis_array(Axis.CARTESIAN_Z)
                zmed = self.coordinates.get_axis_array_med(Axis.CARTESIAN_Z)
                ret_coords = self.coordinates.project_along(
                    Axis.CARTESIAN_Z, zmed[imid].item()
                )
                km = closest_index(zmed, zarr.min())
                kp = closest_index(zmed, zarr.max())
                if z is not None:
                    km = closest_index(zmed, -z)
                    kp = closest_index(zmed, z)
                ret_data = (
                    np.nansum(
                        (self.data * np.ediff1d(zarr))[:, :, km : kp + 1],
                        axis=2,
                        dtype="float64",
                    )
                ).reshape(self.shape[0], self.shape[1], 1)
            case Geometry.POLAR:
                zarr = self.coordinates.get_axis_array(Axis.CARTESIAN_Z)
                zmed = self.coordinates.get_axis_array_med(Axis.CARTESIAN_Z)
                ret_coords = self.coordinates.project_along(
                    Axis.CARTESIAN_Z, zmed[imid].item()
                )
                km = closest_index(zmed, zarr.min())
                kp = closest_index(zmed, zarr.max())
                if z is not None:
                    km = closest_index(zmed, -z)
                    kp = closest_index(zmed, z)
                ret_data = (
                    np.nansum(
                        (self.data * np.ediff1d(zarr))[:, :, km : kp + 1],
                        axis=2,
                        dtype="float64",
                    )
                ).reshape(self.shape[0], self.shape[1], 1)
            case Geometry.SPHERICAL:
                raise NotImplementedError(
                    "vertical_projection(z) function not implemented in spherical coordinates.\n"
                    "Maybe you could use the function latitudinal_projection(theta)?"
                )
            case _ as unreachable:
                assert_never(unreachable)

        return self.replace(
            data=ret_data.astype("float32", copy=False),
            coordinates=ret_coords,
            operation=operation,
        )

    def vertical_at_midplane(
        self,
        *,
        operation_name: str | None = None,
    ) -> "Field[F]":
        operation = self._resolve_operation_name(
            prefix=self.operation,
            default_suffix="vertical_at_midplane",
            operation_name=operation_name,
        )
        imid = self.find_imid()
        match self.native_geometry:
            case Geometry.CARTESIAN:
                zmed = self.coordinates.get_axis_array_med(Axis.CARTESIAN_Z)
                ret_coords = self.coordinates.project_along(
                    Axis.CARTESIAN_Z, zmed[imid].item()
                )
                ret_data = self.data[:, :, imid].reshape(
                    self.shape[0], self.shape[1], 1
                )
                # do geometry conversion!!! -> chainer la conversion (une fois que reduction de dimension -> conversion puis plot egalement chainable)
            case Geometry.POLAR:
                zmed = self.coordinates.get_axis_array_med(Axis.CARTESIAN_Z)
                ret_coords = self.coordinates.project_along(
                    Axis.CARTESIAN_Z, zmed[imid].item()
                )
                ret_data = self.data[:, :, imid].reshape(
                    self.shape[0], self.shape[1], 1
                )
            case Geometry.SPHERICAL:
                thetamed = self.coordinates.get_axis_array_med(Axis.COLATITUDE)
                ret_coords = self.coordinates.project_along(
                    Axis.COLATITUDE, thetamed[imid].item()
                )
                ret_data = self.data[:, imid, :].reshape(
                    self.shape[0], 1, self.shape[2]
                )
            case _ as unreachable:
                assert_never(unreachable)

        return self.replace(
            data=ret_data.astype(self.dtype, copy=False),
            coordinates=ret_coords,
            operation=operation,
        )

    def latitudinal_at_theta(
        self,
        theta: float = 0.0,
        *,
        operation_name: str | None = None,
    ) -> "Field[F]":
        operation = self._resolve_operation_name(
            prefix=self.operation,
            default_suffix=f"latitudinal_at_theta{np.pi / 2 - theta}",
            operation_name=operation_name,
        )

        imid = self.find_imid(altitude=theta)
        match self.native_geometry:
            case Geometry.CARTESIAN:
                raise NotImplementedError(
                    "latitudinal_at_theta is not implemented for cartesian geometry"
                )
            case Geometry.POLAR:
                data_at_theta = np.zeros((self.shape[0], self.shape[1]), dtype=">f4")
                zmed = self.coordinates.get_axis_array_med(Axis.CARTESIAN_Z)
                R = self.coordinates.get_axis_array(Axis.CYLINDRICAL_RADIUS)
                for i in range(self.shape[0]):
                    iz0 = closest_index(zmed, R[i] / np.tan(np.pi / 2 - theta))
                    if np.sign(theta) >= 0:
                        if iz0 < self.shape[2]:
                            data_at_theta[i, :] = self.data[i, :, iz0]
                        else:
                            data_at_theta[i, :] = np.nan
                    else:
                        if iz0 > 0:
                            data_at_theta[i, :] = self.data[i, :, iz0]
                        else:
                            data_at_theta[i, :] = np.nan
                ret_data = data_at_theta.reshape(self.shape[0], self.shape[1], 1)
                ret_coords = self.coordinates.project_along(
                    Axis.CARTESIAN_Z, zmed[imid].item()
                )
            case Geometry.SPHERICAL:
                thetamed = self.coordinates.get_axis_array_med(Axis.COLATITUDE)
                ret_coords = self.coordinates.project_along(
                    Axis.COLATITUDE, thetamed[imid].item()
                )
                ret_data = self.data[
                    :, closest_index(thetamed, np.pi / 2 - theta), :
                ].reshape(self.shape[0], 1, self.shape[2])
            case _ as unreachable:
                assert_never(unreachable)

        return self.replace(
            data=ret_data.astype(self.dtype, copy=False),
            coordinates=ret_coords,
            operation=operation,
        )

    def vertical_at_z(
        self,
        z: float = 0.0,
        *,
        operation_name: str | None = None,
    ) -> "Field[F]":
        operation = self._resolve_operation_name(
            prefix=self.operation,
            default_suffix=f"vertical_at_z{z}",
            operation_name=operation_name,
        )
        imid = self.find_imid(altitude=z)
        match self.native_geometry:
            case Geometry.CARTESIAN:
                zmed = self.coordinates.get_axis_array(Axis.CARTESIAN_Z)
                ret_coords = self.coordinates.project_along(
                    Axis.CARTESIAN_Z, zmed[imid].item()
                )
                ret_data = self.data[:, :, closest_index(zmed, z)].reshape(
                    self.shape[0], self.shape[1], 1
                )
            case Geometry.POLAR:
                zmed = self.coordinates.get_axis_array(Axis.CARTESIAN_Z)
                ret_coords = self.coordinates.project_along(
                    Axis.CARTESIAN_Z, zmed[imid].item()
                )
                ret_data = self.data[:, :, closest_index(zmed, z)].reshape(
                    self.shape[0], self.shape[1], 1
                )
            case Geometry.SPHERICAL:
                raise NotImplementedError(
                    "vertical at z in spherical coordinates not implemented yet."
                )
            case _ as unreachable:
                assert_never(unreachable)

        return self.replace(
            data=ret_data.astype(self.dtype, copy=False),
            coordinates=ret_coords,
            operation=operation,
        )

    def azimuthal_at_phi(
        self,
        phi: float = 0.0,
        *,
        operation_name: str | None = None,
    ) -> "Field[F]":
        operation = self._resolve_operation_name(
            prefix=self.operation,
            default_suffix=f"azimuthal_at_phi{phi}",
            operation_name=operation_name,
        )
        iphi = self.find_iphi(phi=phi)
        match self.native_geometry:
            case Geometry.CARTESIAN:
                raise NotImplementedError(
                    f"geometry flag '{self.native_geometry}' not implemented yet for azimuthal_at_phi"
                )
            case Geometry.POLAR:
                phimed = self.coordinates.get_axis_array(Axis.AZIMUTH)
                ret_coords = self.coordinates.project_along(
                    Axis.AZIMUTH, phimed[iphi].item()
                )
                ret_data = self.data[:, iphi, :].reshape(
                    self.shape[0], 1, self.shape[2]
                )
            case Geometry.SPHERICAL:
                phimed = self.coordinates.get_axis_array(Axis.AZIMUTH)
                ret_coords = self.coordinates.project_along(
                    Axis.AZIMUTH, phimed[iphi].item()
                )
                ret_data = self.data[:, :, iphi].reshape(
                    self.shape[0], self.shape[1], 1
                )
            case _ as unreachable:
                assert_never(unreachable)

        return self.replace(
            data=ret_data.astype(self.dtype, copy=False),
            coordinates=ret_coords,
            operation=operation,
        )

    def azimuthal_at_planet(
        self,
        planet_number: int | None = None,
        *,
        planet_file: str | None = None,
        operation_name: str | None = None,
    ) -> "Field[F]":
        planet_file = _resolve_planet_file(
            planet_number=planet_number, planet_file=planet_file
        )
        del planet_number

        operation = self._resolve_operation_name(
            prefix=self.operation,
            default_suffix="azimuthal_at_planet",
            operation_name=operation_name,
        )

        phip = self.find_phip(planet_file=planet_file)
        aziphip = self.azimuthal_at_phi(phi=phip)
        return aziphip.replace(operation=operation)

    def azimuthal_average(self, *, operation_name: str | None = None) -> "Field[F]":
        operation = self._resolve_operation_name(
            prefix=self.operation,
            default_suffix="azimuthal_average",
            operation_name=operation_name,
        )

        iphi = self.find_iphi(phi=0)
        match self.native_geometry:
            case Geometry.CARTESIAN:
                raise NotImplementedError(
                    f"geometry flag '{self.native_geometry}' not implemented yet for azimuthal_average"
                )
            case Geometry.POLAR:
                phimed = self.coordinates.get_axis_array_med(Axis.AZIMUTH)
                ret_coords = self.coordinates.project_along(
                    Axis.AZIMUTH, phimed[iphi].item()
                )
                ret_data = np.nanmean(self.data, axis=1, dtype="float64").reshape(
                    self.shape[0], 1, self.shape[2]
                )
            case Geometry.SPHERICAL:
                phimed = self.coordinates.get_axis_array_med(Axis.AZIMUTH)
                ret_coords = self.coordinates.project_along(
                    Axis.AZIMUTH, phimed[iphi].item()
                )
                ret_data = np.nanmean(self.data, axis=2, dtype="float64").reshape(
                    self.shape[0], self.shape[1], 1
                )
            case _ as unreachable:
                assert_never(unreachable)

        return self.replace(
            data=ret_data.astype(self.dtype, copy=False),
            coordinates=ret_coords,
            operation=operation,
        )

    def remove_planet_hill_band(
        self,
        planet_number: int | None = None,
        *,
        planet_file: str | None = None,
        operation_name: str | None = None,
    ) -> "Field[F]":
        planet_file = _resolve_planet_file(
            planet_number=planet_number, planet_file=planet_file
        )
        del planet_number

        operation = self._resolve_operation_name(
            prefix=self.operation,
            default_suffix="remove_planet_hill_band",
            operation_name=operation_name,
        )

        phip = self.find_phip(planet_file=planet_file)
        rp = self.find_rp(planet_file=planet_file)
        rhill = self.find_rhill(planet_file=planet_file)
        iphip_m = self.find_iphi(phi=phip - 2 * rhill / rp)
        iphip_p = self.find_iphi(phi=phip + 2 * rhill / rp)
        match self.native_geometry:
            case Geometry.CARTESIAN:
                raise NotImplementedError(
                    f"geometry flag '{self.native_geometry}' not implemented yet for azimuthal_average_except_planet_hill"
                )
            case Geometry.POLAR:
                ret_coords = self.coordinates
                ret_data = self.data.copy()
                if iphip_p >= iphip_m and iphip_p != self.coordinates.shape[1]:
                    ret_data[:, iphip_m : iphip_p + 1, :] = np.nan
                else:
                    if iphip_p == self.coordinates.shape[1]:
                        ret_data[:, iphip_m:iphip_p, :] = np.nan
                    else:
                        ret_data[:, 0 : iphip_p + 1, :] = np.nan
                        ret_data[:, iphip_m : self.coordinates.shape[1], :] = np.nan
            case Geometry.SPHERICAL:
                ret_coords = self.coordinates
                ret_data = self.data.copy()
                if iphip_p >= iphip_m and iphip_p != self.coordinates.shape[2]:
                    ret_data[:, :, iphip_m : iphip_p + 1] = np.nan
                else:
                    if iphip_p == self.coordinates.shape[2]:
                        ret_data[:, :, iphip_m:iphip_p] = np.nan
                    else:
                        ret_data[:, :, 0 : iphip_p + 1] = np.nan
                        ret_data[:, :, iphip_m : self.coordinates.shape[2]] = np.nan
            case _ as unreachable:
                assert_never(unreachable)

        return self.replace(
            data=ret_data.astype(self.dtype, copy=False),
            coordinates=ret_coords,
            operation=operation,
        )

    def radial_at_r(
        self,
        distance: float = 1.0,
        *,
        operation_name: str | None = None,
    ) -> "Field[F]":
        operation = self._resolve_operation_name(
            prefix=self.operation,
            default_suffix=f"radial_at_r{distance}",
            operation_name=operation_name,
        )

        ir1 = self.find_ir(distance=distance)
        match self.native_geometry:
            case Geometry.CARTESIAN:
                raise NotImplementedError(
                    f"geometry flag '{self.native_geometry}' not implemented yet for radial_at_r"
                )
            case Geometry.POLAR:
                rmed = self.coordinates.get_axis_array_med(Axis.CYLINDRICAL_RADIUS)
                ret_coords = self.coordinates.project_along(
                    Axis.CYLINDRICAL_RADIUS, rmed[ir1].item()
                )
            case Geometry.SPHERICAL:
                rmed = self.coordinates.get_axis_array_med(Axis.SPHERICAL_RADIUS)
                ret_coords = self.coordinates.project_along(
                    Axis.SPHERICAL_RADIUS, rmed[ir1].item()
                )
            case _ as unreachable:
                assert_never(unreachable)

        ret_data = self.data[ir1, :, :].reshape(1, self.shape[1], self.shape[2])
        return self.replace(
            data=ret_data.astype(self.dtype, copy=False),
            coordinates=ret_coords,
            operation=operation,
        )

    def radial_average_interval(
        self,
        vmin: float | None = None,
        vmax: float | None = None,
        *,
        operation_name: str | None = None,
    ) -> "Field[F]":
        if (vmin is None) or (vmax is None):
            raise ValueError(
                f"The radial interval {vmin=} and {vmax=} should be defined"
            )

        operation = self._resolve_operation_name(
            prefix=self.operation,
            default_suffix=f"radial_average_interval_{vmin}_{vmax}",
            operation_name=operation_name,
        )

        irmin = self.find_ir(distance=vmin)
        irmax = self.find_ir(distance=vmax)
        ir = self.find_ir(distance=(vmax - vmin) / 2)
        match self.native_geometry:
            case Geometry.CARTESIAN:
                raise NotImplementedError(
                    f"geometry flag '{self.native_geometry}' not implemented yet for radial_at_r"
                )
            case Geometry.POLAR:
                R = self.coordinates.get_axis_array(Axis.CYLINDRICAL_RADIUS)
                if vmin is None:
                    vmin = R.min()
                if vmax is None:
                    vmax = R.max()
                Rmed = self.coordinates.get_axis_array_med(Axis.CYLINDRICAL_RADIUS)
                ret_coords = self.coordinates.project_along(
                    Axis.CYLINDRICAL_RADIUS, Rmed[ir].item()
                )
            case Geometry.SPHERICAL:
                r = self.coordinates.get_axis_array(Axis.SPHERICAL_RADIUS)
                if vmin is None:
                    vmin = r.min()
                if vmax is None:
                    vmax = r.max()
                rmed = self.coordinates.get_axis_array_med(Axis.SPHERICAL_RADIUS)
                ret_coords = self.coordinates.project_along(
                    Axis.SPHERICAL_RADIUS, rmed[ir].item()
                )
            case _ as unreachable:
                assert_never(unreachable)

        ret_data = np.nanmean(
            self.data[irmin : irmax + 1, :, :], axis=0, dtype="float64"
        ).reshape(1, self.shape[1], self.shape[2])
        return self.replace(
            data=ret_data.astype(self.dtype, copy=False),
            coordinates=ret_coords,
            operation=operation,
        )

    def diff(self, on_2: int) -> "Field[F]":
        ds_2 = GasDataSet(
            on_2,
            geometry=self.native_geometry,
            inifile=self.loader.parameter_file,
            directory=self.loader.parameter_file.parent,
        )
        if self.operation != "":
            raise KeyError(
                "For now, diff should only be applied on the initial Field cube."
            )
        ret_data = (self.data - ds_2[self.name].data) / ds_2[self.name].data
        return self.replace(data=ret_data.astype(self.dtype, copy=False))

    def azimuthal_rotation(
        self,
        *,
        rotate_by: float | None = None,
        rotate_with: str | None = None,
        direction: Literal[
            "clockwise",
            "anti-clockwise",
            "trigonometric",
            "anti-trigonometric",
        ] = "clockwise",
    ) -> "Field[F]":
        """
        .. versionadded: 0.20.0
        """
        rotate_by = _resolve_rotate_by(
            rotate_by=rotate_by,
            rotate_with=rotate_with,
            planet_azimuth_finder=partial(
                _find_planet_azimuth,
                loader=self.loader,
                snapshot_uid=self.snapshot_uid,
            ),
        )
        match self.coordinates.geometry:
            case Geometry.POLAR:
                axis = 1
            case Geometry.SPHERICAL:
                axis = 2
            case Geometry.CARTESIAN:
                raise NotImplementedError
            case _ as unreachable:
                assert_never(unreachable)

        match direction:
            case "clockwise" | "anti-trigonometric":
                pass
            case "tigronometric" | "anti-clockwise":
                rotate_by *= -1
            case _:
                raise ValueError(f"Unexpected value {direction=!r}")

        az = self.coordinates.get_axis_array(Axis.AZIMUTH) - rotate_by
        bv = bracketing_values(az, 0)
        v = 2 * np.pi if abs(closest_value(az, 0)) > bv.span else 0.0
        return self.replace(
            data=np.roll(self.data, 1 - closest_index(az, v), axis=axis),
            operation="_".join([self.operation, f"az-shifted-{rotate_by}"]),
        )


class GasField:
    @deprecated(
        "nonos.api.analysis.GasField is deprecated since v0.20.0 "
        "and might be removed in a future version. "
        "In its current state, the class cannot be instantiated: the object created "
        "is already an instance of the replacement class, Field. "
        "Use Field directly to silence this warning."
    )
    def __new__(  # type: ignore[misc]
        cls,
        field: str,
        data: FArray3D[F],
        coords: Coordinates[F],
        ngeom: str,
        on: int,
        operation: str,
        *,
        inifile: os.PathLike[str] | None = None,
        code: str | None = None,
        directory: os.PathLike[str] | None = None,
        rotate_by: float | None = None,
        rotate_with: str | None = None,
    ) -> "Field[F]":
        loader = Loader.resolve(
            code=code,
            parameter_file=inifile,
            directory=Path.cwd() if directory is None else Path(directory),
        )
        field = Field(
            name=field,
            data=data,
            coordinates=coords,
            native_geometry=Geometry(ngeom),
            snapshot_uid=on,
            operation=operation,
            loader=loader,
        )
        if rotate_by or rotate_with:
            return field.azimuthal_rotation(
                rotate_by=rotate_by,
                rotate_with=rotate_with,
            )
        else:
            return field


class GasDataSet(Generic[D, F]):
    """Idefix dataset class that contains everything in the .vtk file

    Args:
        input_dataset (int or str): output number or file name
        directory (str): directory of the .vtk
        geometry (str): for retrocompatibility if old vtk format
        inifile (str): name of the simulation's parameter file if no default files (combined with code)
        code (str): name of the code ("idefix", "pluto", "fargo3d", "fargo-adsg")
    Returns:
        dataset
    """

    def __init__(
        self,
        input_dataset: os.PathLike[str] | int,
        /,
        *,
        inifile: os.PathLike[str] | None = None,
        code: str | None = None,
        geometry: str | None = None,
        directory: os.PathLike[str] | None = None,
        fluid: str | None = None,
        operation: str | None = None,
    ) -> None:
        if isinstance(input_dataset, str | Path):
            input_dataset = Path(input_dataset)
            directory_from_input = input_dataset.parent
            if directory is None:
                directory = directory_from_input
            elif directory_from_input.resolve() != Path(directory).resolve():
                raise ValueError(
                    f"directory value {directory!r} does not match "
                    f"directory name from input_dataset ({directory_from_input!r})"
                )
            del directory_from_input

        if directory is None:
            directory = Path.cwd()
        else:
            directory = Path(directory)

        loader = Loader.resolve(
            code=code,
            parameter_file=inifile,
            directory=directory,
        )

        if fluid is not None and loader.components != BUILTIN_RECIPES["fargo3d"]:
            warnings.warn(
                "Unused keyword argument: 'fluid'",
                category=UserWarning,
                stacklevel=2,
            )

        if operation is not None:
            ignored_kwargs = []
            if fluid is not None:
                ignored_kwargs.append("fluid")
            if geometry is not None:
                ignored_kwargs.append("geometry")
            if ignored_kwargs:
                ignored = ", ".join(repr(_) for _ in ignored_kwargs)
                msg = (
                    "The following keyword arguments are ignored "
                    f"when combined with 'operation': {ignored}"
                )
                warnings.warn(msg, UserWarning, stacklevel=2)
            self._loader = dataclasses.replace(
                loader,
                components=dataclasses.replace(
                    loader.components, binary_reader=NPYReader
                ),
            )
        else:
            self._loader = loader

        self.snapshot_uid, datafile = (
            self._loader.components.binary_reader.parse_snapshot_uid_and_filename(
                input_dataset,
                directory=directory,
                prefix=operation or "",
            )
        )

        self._read = self._loader.load_bin_data(
            datafile,
            geometry=geometry,
            fluid=fluid,
        )

        self._native_geometry = self._read.geometry
        self.coordinates: Coordinates[F] = Coordinates(
            self.native_geometry,
            self._read.x1,
            self._read.x2,
            self._read.x3,
        )
        raw_data: dict[str, FArray3D[F]] = self._read.data
        self.dict: dict[str, Field[F]] = {}
        for key, array in raw_data.items():
            self.dict[key] = Field(
                name=key,
                data=array,
                coordinates=self.coordinates,
                native_geometry=self.native_geometry,
                snapshot_uid=self.snapshot_uid,
                loader=self._loader,
            )

        # backward compatibility for self.params
        self._parameters_input = {
            "inifile": inifile,
            "code": code.removesuffix("_vtk") if code is not None else None,
            "directory": directory,
        }

    @property
    def native_geometry(self) -> Geometry:
        return self._native_geometry

    @property
    @deprecated(
        "GasDataSet.on is deprecated since v0.20.0, "
        "and may be removed in a future version. "
        "Use GasDataSet.snapshot_uid instead"
    )
    def on(self) -> int:
        return self.snapshot_uid

    @property
    @deprecated(
        "GasDataSet.coords is deprecated since v0.20.0, "
        "and may be removed in a future version. "
        "Use GasDataSet.coordinates instead"
    )
    def coords(self) -> Coordinates[F]:
        return self.coordinates

    def __getitem__(self, key: str) -> "Field[F]":
        if key in self.dict:
            return self.dict[key]
        else:
            raise KeyError

    def keys(self) -> KeysView[str]:
        """
        Returns
        =======
        keys of the dict
        """
        return self.dict.keys()

    def values(self) -> ValuesView["Field[F]"]:
        """
        Returns
        =======
        values of the dict
        """
        return self.dict.values()

    def items(self) -> ItemsView[str, "Field[F]"]:
        """
        Returns
        =======
        items of the dict
        """
        return self.dict.items()

    @property
    def nfields(self) -> int:
        """
        Returns
        =======
        The number of fields in the GasDataSet
        """
        return len(self.dict)
