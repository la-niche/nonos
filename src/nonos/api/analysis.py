import dataclasses
import json
import operator as op
import os
import sys
import warnings
from collections import deque
from collections.abc import Callable, ItemsView, KeysView, ValuesView
from dataclasses import dataclass
from functools import partial, wraps
from numbers import Real
from pathlib import Path
from shutil import copyfile
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeAlias,
    TypeVar,
    cast,
    final,
    overload,
)

import numpy as np
from matplotlib.scale import SymmetricalLogTransform
from matplotlib.ticker import SymmetricalLogLocator
from numpy import float32 as f32, float64 as f64

from nonos._geometry import (
    SELECT_ALL,
    AutoIndex,
    Axis,
    Coordinates,
    Geometry,
    IndexRange,
    axes_from_geometry,
)
from nonos._integrity_checks import (
    check_field_operands,
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
    _fequal,
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
        **kwargs: Unpack[PColorMeshKwargs],
    ) -> "Artist":
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


FieldOp: TypeAlias = Callable[["Field[F]", Any], "Field[F]"]


def _arithmetic_field_operator(
    baseop: Callable[[FArray3D[F], Any], FArray3D[F]],
    result_name: str,
    *,
    reverse_operands: bool = False,
) -> Callable[["FieldOp[F]"], "FieldOp[F]"]:
    def decorator(meth: "FieldOp[F]") -> "FieldOp[F]":
        @wraps(meth)
        def impl(f1: "Field[F]", f2: Any) -> "Field[F]":
            operands: tuple[FArray3D[F], Any]
            match f2:
                case Field():
                    if excs := check_field_operands(f1, f2):
                        raise excs
                    f2 = cast("Field[F]", f2)
                    operands = (f1.data, f2.data)
                case Real():
                    operands = (f1.data, f2)
                case _:
                    return NotImplemented  # type: ignore[no-any-return]
            if reverse_operands:
                operands = tuple(reversed(operands))
            return f1.replace(name=result_name, data=baseop(*operands).astype(f1.dtype))

        return impl

    return decorator


class FieldAttrs(Generic[F], TypedDict, total=False):
    name: str
    data: FArray3D[F]
    coordinates: Coordinates[F]


T = TypeVar("T", f32, f64)


@final
@dataclass(slots=True, frozen=True, kw_only=True)
class Field(Generic[F]):
    name: str
    data: FArray3D[F]
    coordinates: Coordinates[F]

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
    def geometry(self) -> Geometry:  # pragma: no cover
        return self.coordinates.geometry

    def replace(self, **subs: Unpack[FieldAttrs[F]]) -> "Field[F]":
        """Convenience wrapper around copy.replace"""
        if sys.version_info >= (3, 13):
            from copy import replace
        else:
            from dataclasses import replace
        return replace(self, **subs)

    @property
    def dtype(self) -> np.dtype[F]:  # pragma: no cover
        return self.coordinates.dtype

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

    def astype(self, dtype: np.dtype[T]) -> "Field[T]":
        """
        For convenience, mimic np.ndarray.astype

        The underlying data is always copied.

        .. versionadded: 0.20.0
        """
        return Field(
            name=self.name,
            data=self.data.astype(dtype),
            coordinates=self.coordinates.astype(dtype),
        )

    def __eq__(self, other: Any) -> bool:
        # checks are ordered from cheapest to most expensive
        # so False is returned as early as possible when other
        # is not comparable
        return other is self or (
            type(other) is Field
            and self.name == other.name
            and self.coordinates == other.coordinates
            and bool(other.data is self.data or np.all(other.data == self.data))
        )

    # low level arithmetic

    # despite my best effort, type checkers (mypy and ty) do not
    # seem able to infer that these decorated methods are in fact
    # type-safe: their real bodies live in the decorator's implementation
    @_arithmetic_field_operator(op.add, "<sum-result>")
    def __add__(self, other: Any) -> "Field[F]": ...  # type: ignore
    @_arithmetic_field_operator(op.add, "<sum-result>", reverse_operands=True)
    def __radd__(self, other: Any) -> "Field[F]": ...  # type: ignore
    @_arithmetic_field_operator(op.sub, "<sub-result>")
    def __sub__(self, other: Any) -> "Field[F]": ...  # type: ignore
    @_arithmetic_field_operator(op.sub, "<sub-result>", reverse_operands=True)
    def __rsub__(self, other: Any) -> "Field[F]": ...  # type: ignore
    @_arithmetic_field_operator(op.mul, "<mul-result>")
    def __mul__(self, other: Any) -> "Field[F]": ...  # type: ignore
    @_arithmetic_field_operator(op.mul, "<mul-result>", reverse_operands=True)
    def __rmul__(self, other: Any) -> "Field[F]": ...  # type: ignore
    @_arithmetic_field_operator(op.truediv, "<truediv-result>")
    def __truediv__(self, other: Any) -> "Field[F]": ...  # type: ignore
    @_arithmetic_field_operator(op.truediv, "<truediv-result>", reverse_operands=True)
    def __rtruediv__(self, other: Any) -> "Field[F]": ...  # type: ignore

    # medium level methods:
    # - an axis must always be specified
    # - any axis from the field's geometry is considered valid, all others are invalid
    # - indices (deltas) must be provided as exact integer values
    # - output names should reflect the operation conducted (e.g. <slice-result>)

    # reductions: methods that reduce effective_ndim
    # these should all be idempotents
    def slice_at_index(
        self,
        axis: Axis,
        idx: int | AutoIndex,
        /,
        *,
        keep_name: bool = False,
    ) -> "Field[F]":
        sel: list[slice[None] | int] = [slice(None), slice(None), slice(None)]
        axis_idx = self.coordinates.get_axis_index(axis)
        match idx:
            case AutoIndex.LEFTMOST:
                idx = 0
            case AutoIndex.RIGHTMOST:
                idx = -1
            case AutoIndex.MIDPOINT:
                idx = self.shape[axis_idx] // 2

        sel[axis_idx] = idx
        new_shape = list(self.shape)
        new_shape[axis_idx] = 1

        return Field(
            name=self.name if keep_name else "<slice-result>",
            data=self.data[tuple(sel)].reshape(new_shape),  # type: ignore[arg-type]
            coordinates=self.coordinates.slice_at_index(axis, idx),
        )

    def select_subdomain(
        self,
        selection: dict[Axis, IndexRange],
        /,
        *,
        keep_name: bool = False,
    ) -> "Field[F]":
        if len(selection) > 3:
            raise TypeError

        selection = (
            dict.fromkeys(axes_from_geometry(self.geometry), SELECT_ALL) | selection
        )
        ranges = cast(
            tuple[IndexRange, IndexRange, IndexRange], tuple(selection.values())
        )
        sel = cast(tuple[slice, slice, slice], tuple(ir.as_slice() for ir in ranges))

        return Field(
            name=self.name if keep_name else "<select-result>",
            data=self.data[sel],
            coordinates=self.coordinates.select_logical_slab(
                x1=ranges[0], x2=ranges[1], x3=ranges[2]
            ),
        )


class GasFieldReplaceKwargs(Generic[F], TypedDict, total=False):
    _field: Field[F]
    name: str
    data: FArray3D[F]
    coordinates: Coordinates[F]
    snapshot_uid: int
    operation: str
    rotate_by: float


@final
class GasField(Generic[F]):
    def __init__(
        self,
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
    ) -> None:
        if ngeom != coords.geometry:
            raise ValueError

        self._field: Field[F] = Field(
            name=field,
            data=data,
            coordinates=coords,
        )

        self._snapshot_uid = on
        self._operation = operation
        self._loader = Loader.resolve(
            code=code,
            parameter_file=inifile,
            directory=Path.cwd() if directory is None else Path(directory),
        )
        self._rotate_by = _resolve_rotate_by(
            rotate_by=rotate_by,
            rotate_with=rotate_with,
            planet_azimuth_finder=partial(
                _find_planet_azimuth,
                loader=self._loader,
                snapshot_uid=on,
            ),
        )

    @property
    def snapshot_uid(self) -> int:
        return self._snapshot_uid

    @property
    def snapshot_number(self) -> int:  # pragma: no cover
        # supported (non-deprecated) alias
        return self.snapshot_uid

    @property
    def directory(self) -> Path:
        return self._loader.parameter_file.parent

    @property
    def operation(self) -> str:
        return self._operation

    # thinly wrap the underlying Field object by re-exposing its attributes
    # as read-only properties
    @property
    def name(self) -> str:
        return self._field.name

    @property
    def data(self) -> FArray3D[F]:
        return self._field.data

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
        return self._field.as_ndview(ndim=ndim)

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

    @property
    def coordinates(self) -> Coordinates[F]:
        return self._field.coordinates

    @property
    def dtype(self) -> np.dtype[F]:
        return self._field.dtype

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._field.shape

    @property
    def effective_ndim(self) -> Literal[0, 1, 2, 3]:
        """
        The effective dimensionality of the underlying data.
        This corresponds to the number of dimensions with more than a single element.

        .. versionadded: 0.20.0
        """
        return self._field.effective_ndim

    @property
    def geometry(self) -> Geometry:
        return self.coordinates.geometry

    def replace(self, **subs: Unpack[GasFieldReplaceKwargs[F]]) -> "GasField[F]":
        """
        .. versionadded: 0.20.0
        """

        new = object.__new__(GasField)

        field_kws: FieldAttrs[F] = {}
        subs_cpy = subs.copy()
        for k in ["name", "data", "coordinates"]:
            if k in subs_cpy:
                field_kws[k] = subs.pop(k)  # type: ignore[literal-required, misc]
        if field_kws:
            if "_field" in subs:
                raise TypeError
            new._field = self._field.replace(**field_kws)
        else:
            new._field = subs.get("_field", self._field)

        new._loader = self._loader
        new._snapshot_uid = subs.get("snapshot_uid", self._snapshot_uid)
        new._operation = subs.get("operation", self._operation)
        new._rotate_by = subs.get("rotate_by", self._rotate_by)
        return new

    # legacy, deprecated aliases
    @property
    @deprecated(
        "GasField.field is deprecated since v0.20.0, "
        "and may be removed in a future version. "
        "Use GasField.name instead"
    )
    def field(self) -> str:  # pragma: no cover
        return self.name

    @property
    @deprecated(
        "GasField.inifile is deprecated since v0.20.0, "
        "and may be removed in a future version."
    )
    def inifile(self) -> Path:  # pragma: no cover
        return self._loader.parameter_file

    @property
    @deprecated(
        "GasField.coords is deprecated since v0.20.0, "
        "and may be removed in a future version. "
        "Use GasField.coordinates instead"
    )
    def coords(self) -> Coordinates[F]:
        return self.coordinates

    @property
    @deprecated(
        "GasField.on is deprecated since v0.20.0, "
        "and may be removed in a future version. "
        "Use GasField.snapshot_uid instead"
    )
    def on(self) -> int:  # pragma: no cover
        return self.snapshot_uid

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
        rotate_by=None,
        rotate_with=None,
    ):
        rotate_by = _resolve_rotate_by(
            rotate_by=rotate_by,
            rotate_with=rotate_with,
            planet_azimuth_finder=partial(
                _find_planet_azimuth,
                loader=self._loader,
                snapshot_uid=self.snapshot_uid,
            ),
        )

        data_key = self.name
        if self.effective_ndim > 2:
            raise ValueError("data has to be 1D or 2D in order to call map.")

        axis_1 = Axis.from_label(a)

        if b is None:
            mesh1D = self.coordinates._meshgrid_conversion_1d(axis_1)

            abscissa_val1D = next(iter(mesh1D.values()))
            abscissa_key = next(iter(mesh1D.keys()))
            if axis_1 is Axis.AZIMUTH and not _fequal(self._rotate_by, rotate_by):
                phicoord = self.coordinates.get_axis_array(Axis.AZIMUTH) - rotate_by
                bv = bracketing_values(phicoord, 0)
                if abs(closest_value(phicoord, 0)) > bv.span:
                    ipi = closest_index(phicoord, 2 * np.pi)
                else:
                    ipi = closest_index(phicoord, 0)
                match self.geometry:
                    case Geometry.POLAR:
                        data_view = np.roll(self.data, -ipi + 1, axis=1)
                    case Geometry.SPHERICAL:
                        data_view = np.roll(self.data, -ipi + 1, axis=2)
                    case _:
                        raise NotImplementedError(
                            f"geometry flag '{self.geometry}' not implemented yet if corotation"
                        )
            else:
                data_view = self.data.view()

            return Plotable(
                abscissa=(abscissa_key.label, abscissa_val1D),
                ordinate=(data_key, data_view.squeeze()),
            )

        else:
            axis_2 = Axis.from_label(b)

            mesh2D = self.coordinates._meshgrid_conversion_2d(axis_1, axis_2)
            # meshgrid in polar coordinates P, R (if "R", "phi") or R, P (if "phi", "R")
            # idem for all combinations of R,phi,z
            abscissa_val2D, ordinate_value = (mesh2D[axis_1], mesh2D[axis_2])
            abscissa_key, ordinate_key = (axis_1, axis_2)
            native_plane_axes = self.coordinates.native_from_wanted(axis_1, axis_2)
            if Axis.AZIMUTH in native_plane_axes and not _fequal(
                self._rotate_by, rotate_by
            ):
                phicoord = self.coordinates.get_axis_array(Axis.AZIMUTH) - rotate_by
                bv = bracketing_values(phicoord, 0)
                if abs(closest_value(phicoord, 0)) > bv.span:
                    ipi = closest_index(phicoord, 2 * np.pi)
                else:
                    ipi = closest_index(phicoord, 0)
                match self.geometry:
                    case Geometry.POLAR:
                        data_view = np.roll(self.data, -ipi + 1, axis=1)
                    case Geometry.SPHERICAL:
                        data_view = np.roll(self.data, -ipi + 1, axis=2)
                    case _:
                        raise NotImplementedError(
                            f"geometry flag '{self.geometry}' not implemented yet if corotation"
                        )
            else:
                data_view = self.data.view()

            def rotate_axes(arr: FArray[D, F], shift: int) -> FArray[D, F]:
                axes_in = tuple(range(arr.ndim))
                axes_out = deque(axes_in)
                axes_out.rotate(shift)
                return np.moveaxis(arr, axes_in, axes_out)

            # make reduction axis the first axis then drop (squeeze) it,
            # while preserving the original (cyclic) order in the other two axes
            data_view = rotate_axes(data_view, shift=self.shape.index(1)).squeeze()

            naxes = axes_from_geometry(self.geometry)
            sorted_pairs = [
                (naxes[0], naxes[1]),
                (naxes[1], naxes[2]),
                (naxes[2], naxes[0]),
            ]
            if native_plane_axes in sorted_pairs:
                data_view = data_view.T

            return Plotable(
                abscissa=(abscissa_key.label, abscissa_val2D),
                ordinate=(ordinate_key.label, ordinate_value),
                field=(data_key, data_view),
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
        operation = self._operation
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

        src = self._loader.parameter_file
        dest = directory / self._loader.parameter_file.name
        if dest != src:
            copyfile(src, dest)

        return file

    def find_ir(self, distance: float = 1.0) -> int:
        match self.geometry:
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

    def find_ialt(self, altitude: float, /) -> int:
        match self.geometry:
            case Geometry.CARTESIAN | Geometry.POLAR:
                arr = self.coordinates.get_axis_array_med(Axis.CARTESIAN_Z)
                return closest_index(arr, altitude)
            case Geometry.SPHERICAL:
                arr = self.coordinates.get_axis_array_med(Axis.COLATITUDE)
                return closest_index(arr, np.pi / 2 - altitude)
            case _ as unreachable:
                assert_never(unreachable)

    @deprecated(
        "GasField.find_imid is deprecated since v0.21.0 and "
        "might be removed in a future version. Use GasField.find_ialt instead."
    )
    def find_imid(self, altitude: float = 0.0) -> int:  # pragma: no cover
        return self.find_ialt(altitude)

    def find_iphi(self, phi: float = 0) -> int:
        match self.geometry:
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
        file = self._loader.parameter_file.parent / planet_file
        return self._loader.load_planet_data(file)

    def _get_ind_snapshot_uid(self, time: FArray1D[F]) -> int:
        return _get_ind_snapshot_uid(self._loader, self.snapshot_uid, time)

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
        ini = self._loader.load_ini_file()
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
            self._loader,
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
    ) -> "GasField[F]":
        default_suffix = "latitudinal_projection"
        if theta is not None:
            default_suffix += str(np.pi / 2 - theta)
        operation = self._resolve_operation_name(
            prefix=self._operation,
            default_suffix=default_suffix,
            operation_name=operation_name,
        )

        imid = self.find_ialt(0.0)
        match self.geometry:
            case Geometry.CARTESIAN:
                raise NotImplementedError(
                    "latitudinal_projection isn't implemented for cartesian geometry"
                )
            case Geometry.POLAR:
                ret_coords = Coordinates(
                    self.geometry,
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
                    self.geometry,
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
    ) -> "GasField[F]":
        imid = self.find_ialt(0.0)
        match self.geometry:
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
            operation=self._resolve_operation_name(
                prefix=self._operation,
                default_suffix=f"vertical_projection{z if z is not None else ''}",
                operation_name=operation_name,
            ),
        )

    def vertical_at_midplane(
        self,
        *,
        operation_name: str | None = None,
    ) -> "GasField[F]":
        match self.geometry:
            case Geometry.CARTESIAN | Geometry.POLAR:
                axis = Axis.CARTESIAN_Z
            case Geometry.SPHERICAL:
                axis = Axis.COLATITUDE
            case _ as unreachable:
                assert_never(unreachable)

        return self.replace(
            _field=self._field.slice_at_index(
                axis, self.find_ialt(0.0), keep_name=True
            ),
            operation=self._resolve_operation_name(
                prefix=self._operation,
                default_suffix="vertical_at_midplane",
                operation_name=operation_name,
            ),
        )

    def latitudinal_at_theta(
        self,
        theta: float = 0.0,
        *,
        operation_name: str | None = None,
    ) -> "GasField[F]":
        imid = self.find_ialt(theta)
        match self.geometry:
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
            case Geometry.CARTESIAN:
                raise NotImplementedError
            case _ as unreachable:
                assert_never(unreachable)

        return self.replace(
            data=ret_data.astype(self.dtype, copy=False),
            coordinates=ret_coords,
            operation=self._resolve_operation_name(
                prefix=self._operation,
                default_suffix=f"latitudinal_at_theta{np.pi / 2 - theta}",
                operation_name=operation_name,
            ),
        )

    def vertical_at_z(
        self,
        z: float = 0.0,
        *,
        operation_name: str | None = None,
    ) -> "GasField[F]":
        match self.geometry:
            case Geometry.CARTESIAN | Geometry.POLAR:
                axis = Axis.CARTESIAN_Z
            case Geometry.SPHERICAL:
                raise NotImplementedError
            case _ as unreachable:
                assert_never(unreachable)

        return self.replace(
            _field=self._field.slice_at_index(axis, self.find_ialt(z), keep_name=True),
            operation=self._resolve_operation_name(
                prefix=self._operation,
                default_suffix=f"vertical_at_z{z}",
                operation_name=operation_name,
            ),
        )

    def azimuthal_at_phi(
        self,
        phi: float = 0.0,
        *,
        operation_name: str | None = None,
    ) -> "GasField[F]":
        match self.geometry:
            case Geometry.POLAR | Geometry.SPHERICAL:
                axis = Axis.AZIMUTH
            case Geometry.CARTESIAN:
                raise NotImplementedError
            case _ as unreachable:
                assert_never(unreachable)

        return self.replace(
            _field=self._field.slice_at_index(
                axis, self.find_iphi(phi=phi), keep_name=True
            ),
            operation=self._resolve_operation_name(
                prefix=self._operation,
                default_suffix=f"azimuthal_at_phi{phi}",
                operation_name=operation_name,
            ),
        )

    def azimuthal_at_planet(
        self,
        planet_number: int | None = None,
        *,
        planet_file: str | None = None,
        operation_name: str | None = None,
    ) -> "GasField[F]":
        return self.azimuthal_at_phi(
            phi=self.find_phip(
                planet_file=_resolve_planet_file(
                    planet_number=planet_number,
                    planet_file=planet_file,
                )
            )
        ).replace(
            operation=self._resolve_operation_name(
                prefix=self._operation,
                default_suffix="azimuthal_at_planet",
                operation_name=operation_name,
            )
        )

    def azimuthal_average(self, *, operation_name: str | None = None) -> "GasField[F]":
        iphi = self.find_iphi(phi=0)
        match self.geometry:
            case Geometry.CARTESIAN:
                raise NotImplementedError(
                    f"geometry flag '{self.geometry}' not implemented yet for azimuthal_average"
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
            operation=self._resolve_operation_name(
                prefix=self._operation,
                default_suffix="azimuthal_average",
                operation_name=operation_name,
            ),
        )

    def remove_planet_hill_band(
        self,
        planet_number: int | None = None,
        *,
        planet_file: str | None = None,
        operation_name: str | None = None,
    ) -> "GasField[F]":
        planet_file = _resolve_planet_file(
            planet_number=planet_number, planet_file=planet_file
        )
        del planet_number

        phip = self.find_phip(planet_file=planet_file)
        rp = self.find_rp(planet_file=planet_file)
        rhill = self.find_rhill(planet_file=planet_file)
        iphip_m = self.find_iphi(phi=phip - 2 * rhill / rp)
        iphip_p = self.find_iphi(phi=phip + 2 * rhill / rp)
        match self.geometry:
            case Geometry.CARTESIAN:
                raise NotImplementedError(
                    f"geometry flag '{self.geometry}' not implemented yet for azimuthal_average_except_planet_hill"
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
            operation=self._resolve_operation_name(
                prefix=self._operation,
                default_suffix="remove_planet_hill_band",
                operation_name=operation_name,
            ),
        )

    def radial_at_r(
        self,
        distance: float = 1.0,
        *,
        operation_name: str | None = None,
    ) -> "GasField[F]":
        match self.geometry:
            case Geometry.POLAR:
                axis = Axis.CYLINDRICAL_RADIUS
            case Geometry.SPHERICAL:
                axis = Axis.SPHERICAL_RADIUS
            case Geometry.CARTESIAN:
                raise NotImplementedError
            case _ as unreachable:
                assert_never(unreachable)

        return self.replace(
            _field=self._field.slice_at_index(
                axis, self.find_ir(distance=distance), keep_name=True
            ),
            operation=self._resolve_operation_name(
                prefix=self._operation,
                default_suffix=f"radial_at_r{distance}",
                operation_name=operation_name,
            ),
        )

    def radial_average_interval(
        self,
        vmin: float,
        vmax: float,
        *,
        operation_name: str | None = None,
    ) -> "GasField[F]":
        irmin = self.find_ir(distance=vmin)
        irmax = self.find_ir(distance=vmax)
        ir = self.find_ir(distance=(vmax - vmin) / 2)
        match self.geometry:
            case Geometry.CARTESIAN:
                raise NotImplementedError(
                    f"geometry flag '{self.geometry}' not implemented yet for radial_at_r"
                )
            case Geometry.POLAR:
                Rmed = self.coordinates.get_axis_array_med(Axis.CYLINDRICAL_RADIUS)
                ret_coords = self.coordinates.project_along(
                    Axis.CYLINDRICAL_RADIUS, Rmed[ir].item()
                )
            case Geometry.SPHERICAL:
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
            operation=self._resolve_operation_name(
                prefix=self._operation,
                default_suffix=f"radial_average_interval_{vmin}_{vmax}",
                operation_name=operation_name,
            ),
        )

    def diff(self, on_2: int) -> "GasField[F]":
        ds_2 = GasDataSet(
            on_2,
            geometry=self.geometry,
            inifile=self._loader.parameter_file,
            directory=self._loader.parameter_file.parent,
        )
        if self._operation != "":
            raise KeyError(
                "For now, diff should only be applied on the initial Field cube."
            )
        gf = ds_2[self.name]
        ret_data = ((self._field - gf._field) / gf._field).data
        return self.replace(data=ret_data.astype(self.dtype, copy=False))

    def rotate(
        self,
        *,
        rotate_with: str | None = None,
        rotate_by: float | None = None,
    ) -> "GasField[F]":
        rotate_by = _resolve_rotate_by(
            rotate_by=rotate_by,
            rotate_with=rotate_with,
            planet_azimuth_finder=partial(
                _find_planet_azimuth,
                loader=self._loader,
                snapshot_uid=self.snapshot_uid,
            ),
        )
        if self.effective_ndim < 2:
            raise ValueError("data has to be 2D or 3D in order to rotate the data.")
        if not _fequal(self._rotate_by, rotate_by):
            phicoord = self.coordinates.get_axis_array(Axis.AZIMUTH) - rotate_by
            if abs(closest_value(phicoord, 0)) > bracketing_values(phicoord, 0).span:
                ipi = closest_index(phicoord, 2 * np.pi)
            else:
                ipi = closest_index(phicoord, 0)
            match self.geometry:
                case Geometry.POLAR:
                    ret_data = np.roll(self.data, -ipi + 1, axis=1)
                case Geometry.SPHERICAL:
                    ret_data = np.roll(self.data, -ipi + 1, axis=2)
                case _:
                    raise NotImplementedError(
                        f"geometry flag '{self.geometry}' not implemented yet if corotation"
                    )
        else:
            ret_data = self.data

        return self.replace(data=ret_data.astype(self.dtype, copy=False))


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

        bd = self._loader.load_bin_data(datafile, geometry=geometry, fluid=fluid)

        self._coordinates = Coordinates(
            bd.geometry,
            bd.x1,
            bd.x2,
            bd.x3,
        )
        self.dict: dict[str, GasField[F]] = {}
        for key, array in bd.data.items():
            self.dict[key] = GasField(
                key,
                data=array,
                coords=self.coordinates,
                ngeom=self.coordinates.geometry,
                on=self.snapshot_uid,
                inifile=self._loader.parameter_file,
                operation="",
            )

        # backward compatibility for self.params
        self._parameters_input = {
            "inifile": inifile,
            "code": code.removesuffix("_vtk") if code is not None else None,
            "directory": directory,
        }

    @property
    def coordinates(self) -> Coordinates[F]:
        return cast(Coordinates[F], self._coordinates)

    @property
    def geometry(self) -> Geometry:
        return self._coordinates.geometry

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

    @property
    @deprecated(
        "GasDataSet.native_geometry is deprecated since v0.20.0, "
        "and may be removed in a future version. "
        "Use GasDataSet.geometry instead"
    )
    def native_geometry(self) -> Geometry:
        return self.geometry

    def __getitem__(self, key: str) -> "GasField[F]":
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

    def values(self) -> ValuesView["GasField[F]"]:
        """
        Returns
        =======
        values of the dict
        """
        return self.dict.values()

    def items(self) -> ItemsView[str, "GasField[F]"]:
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
