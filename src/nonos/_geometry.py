from __future__ import annotations

__all__ = [
    "SELECT_ALL",
    "AutoIndex",
    "Axis",
    "Coordinates",
    "Geometry",
    "IndexRange",
    "axes_from_geometry",
]
import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Generic, Literal, TypeVar, cast, final, overload

import numpy as np
from numpy import float32 as f32, float64 as f64

from nonos._integrity_checks import collect_dtype_exceptions, compile_exceptions
from nonos._types import D, F, FArray, FArray1D, FArray2D, StrDict

if sys.version_info >= (3, 11):
    from enum import StrEnum
    from typing import assert_never
else:
    from typing_extensions import assert_never

    from nonos._backports import StrEnum
if sys.version_info >= (3, 13):
    from copy import replace
    from warnings import deprecated
else:
    from dataclasses import replace

    from typing_extensions import deprecated


class Geometry(StrEnum):
    CARTESIAN = auto()
    POLAR = auto()
    SPHERICAL = auto()


class Axis(Enum):
    CARTESIAN_X = auto()
    CARTESIAN_Y = auto()
    CARTESIAN_Z = auto()
    AZIMUTH = auto()
    CYLINDRICAL_RADIUS = auto()
    SPHERICAL_RADIUS = auto()
    COLATITUDE = auto()

    @property
    def label(self) -> str:
        return _AXIS_TO_STR[self]

    @classmethod
    def from_label(cls, label: str) -> Axis:
        return _STR_TO_AXIS[label]


_AXIS_TO_STR = {
    Axis.CARTESIAN_X: "x",
    Axis.CARTESIAN_Y: "y",
    Axis.CARTESIAN_Z: "z",
    Axis.AZIMUTH: "phi",
    Axis.CYLINDRICAL_RADIUS: "R",
    Axis.SPHERICAL_RADIUS: "r",
    Axis.COLATITUDE: "theta",
}
_STR_TO_AXIS = {v: k for k, v in _AXIS_TO_STR.items()}
assert all(axis in _AXIS_TO_STR for axis in Axis)


class AutoIndex(Enum):
    LEFTMOST = auto()
    RIGHTMOST = auto()
    MIDPOINT = auto()


@dataclass(slots=True, frozen=True)
class IndexRange:
    start: int | None = None
    stop: int | None = None

    def __post_init__(self) -> None:
        if (stop := self.stop) is not None and stop <= 0:
            raise ValueError(f"Expected stop>0, got {stop=}")

        if (start := self.start) is not None and start < 0:
            raise ValueError(f"Expected start>=0, got {start=}")

        if start is not None and stop is not None and stop < start:
            raise ValueError(f"Expected start<stop, got {start=}, {stop=}")

    def as_slice(self) -> slice:
        return slice(self.start, self.stop)


SELECT_ALL = IndexRange()


def axes_from_geometry(geometry: Geometry, /) -> tuple[Axis, Axis, Axis]:
    match geometry:
        case Geometry.CARTESIAN:
            return (Axis.CARTESIAN_X, Axis.CARTESIAN_Y, Axis.CARTESIAN_Z)
        case Geometry.POLAR:
            return (Axis.CYLINDRICAL_RADIUS, Axis.AZIMUTH, Axis.CARTESIAN_Z)
        case Geometry.SPHERICAL:
            return (Axis.SPHERICAL_RADIUS, Axis.COLATITUDE, Axis.AZIMUTH)
        case _ as unreachable:
            assert_never(unreachable)


def _get_target_geometry(*axes: Axis) -> Geometry:
    axes_set = set(axes)
    if len(axes_set) == 0 or len(axes_set) > 3:
        raise AssertionError

    candidates: set[Geometry] = set()
    for geom in Geometry:
        if set(axes_from_geometry(geom)).issuperset(axes_set):
            candidates.add(geom)

    if len(candidates) == 0:
        raise AssertionError
    elif len(candidates) > 1:
        # supposedly unreachable
        raise AssertionError

    return candidates.pop()


def _native_axis_from_target_axis(
    native_geometry: Geometry,
    axis: Axis,
) -> Axis:
    if axis in axes_from_geometry(native_geometry):
        return axis

    match native_geometry, axis:
        # SPHERICAL to CARTESIAN
        case Geometry.SPHERICAL, Axis.CARTESIAN_X:
            return Axis.SPHERICAL_RADIUS
        case Geometry.SPHERICAL, Axis.CARTESIAN_Y:
            return Axis.AZIMUTH  # grotesque
        case Geometry.SPHERICAL, Axis.CARTESIAN_Y:
            return Axis.COLATITUDE

        # SPHERICAL to POLAR
        case Geometry.SPHERICAL, Axis.CYLINDRICAL_RADIUS:
            return Axis.SPHERICAL_RADIUS

        # POLAR to CARTESIAN
        case Geometry.POLAR, Axis.CARTESIAN_X:
            return Axis.CYLINDRICAL_RADIUS
        case Geometry.POLAR, Axis.CARTESIAN_Y:
            return Axis.AZIMUTH  # grotesque

        # POLAR to SPHERICAL
        case Geometry.POLAR, Axis.SPHERICAL_RADIUS:
            return Axis.CYLINDRICAL_RADIUS
        case Geometry.POLAR, Axis.COLATITUDE:
            return Axis.CARTESIAN_Z

        case _:
            raise NotImplementedError(
                f"Transformation from {native_geometry} to {axis} is not implemented"
            )


def _native_plane_from_target_plane(
    native_geometry: Geometry,
    axis_1: Axis,
    axis_2: Axis,
    _recurse: bool = True,
) -> tuple[Axis, Axis]:
    # this replaces Coordinates.native_from_wanted to a certain extent:
    # - I only care about 2D cases
    # - I assume input axis are already validated
    #   (i.e. are members from a uniquely defined geometry)
    #
    # I'll start with a bug-for-bug approach, but I believe that most cases
    #   are actually broken:
    #
    # There are 3 geometries to convert to and from, all with 3 axes.
    # Choosing a target plane requires 2 axes, which gives 3 possible planes
    # per target geometry => 3^3 = 27 (order insensitive) cases
    # I use the following categories:
    #   * native: trivially correct (transform is identity) (9/27)
    #   * not implemented: cartesian to anything different isn't supported (6/27)
    #   * genuine: unambiguously correct (3/27)
    #   * no comment: not 100% correct but arguably useful (7/27)
    #   * grotesque: pretty sure this is never wanted (2/27)
    #
    # "native" and "genuine" cases are the only categories where no information
    # is lost in conversion. In the general case we need all 3 axes from native
    # geometry to project to a well-defined plane in another geometry.

    input_tuple = (axis_1, axis_2)
    if set(input_tuple).issubset(axes_from_geometry(native_geometry)):
        return input_tuple

    match native_geometry, input_tuple:
        # SPHERICAL to CARTESIAN
        case Geometry.SPHERICAL, (Axis.CARTESIAN_X, Axis.CARTESIAN_Y):
            return (Axis.SPHERICAL_RADIUS, Axis.AZIMUTH)
        case Geometry.SPHERICAL, (Axis.CARTESIAN_X, Axis.CARTESIAN_Z):
            return (Axis.SPHERICAL_RADIUS, Axis.COLATITUDE)
        case Geometry.SPHERICAL, (Axis.CARTESIAN_Y, Axis.CARTESIAN_Z):
            return (Axis.AZIMUTH, Axis.COLATITUDE)  # grotesque

        # SPHERICAL to POLAR
        case Geometry.SPHERICAL, (Axis.CYLINDRICAL_RADIUS, Axis.AZIMUTH):
            return (Axis.SPHERICAL_RADIUS, Axis.AZIMUTH)
        case Geometry.SPHERICAL, (Axis.CYLINDRICAL_RADIUS, Axis.CARTESIAN_Z):
            return (Axis.SPHERICAL_RADIUS, Axis.COLATITUDE)  # genuine
        case Geometry.SPHERICAL, (Axis.AZIMUTH, Axis.CARTESIAN_Z):
            return (Axis.AZIMUTH, Axis.COLATITUDE)

        # POLAR to CARTESIAN
        case Geometry.POLAR, (Axis.CARTESIAN_X, Axis.CARTESIAN_Y):
            return (Axis.CYLINDRICAL_RADIUS, Axis.AZIMUTH)  # genuine
        case Geometry.POLAR, (Axis.CARTESIAN_X, Axis.CARTESIAN_Z):
            return (Axis.CYLINDRICAL_RADIUS, Axis.CARTESIAN_Z)
        case Geometry.POLAR, (Axis.CARTESIAN_Y, Axis.CARTESIAN_Z):
            return (Axis.AZIMUTH, Axis.CARTESIAN_Z)  # grotesque

        # POLAR to SPHERICAL
        case Geometry.POLAR, (Axis.SPHERICAL_RADIUS, Axis.AZIMUTH):
            return (Axis.CYLINDRICAL_RADIUS, Axis.AZIMUTH)
        case Geometry.POLAR, (Axis.SPHERICAL_RADIUS, Axis.COLATITUDE):
            return (Axis.CYLINDRICAL_RADIUS, Axis.CARTESIAN_Z)  # genuine
        case Geometry.POLAR, (Axis.AZIMUTH, Axis.COLATITUDE):
            return (Axis.AZIMUTH, Axis.CARTESIAN_Z)

        case _ if _recurse:
            # order matters. Rercursion allows use to implement only half of all cases
            a1, a2 = _native_plane_from_target_plane(
                native_geometry,
                axis_2,
                axis_1,
                _recurse=False,
            )
            return a2, a1

        case _:
            # note that we can only land here after recursion/input inversion,
            # so we revert the order of arguments back to what was passed from
            # the outside
            raise NotImplementedError(
                f"Transformation from {native_geometry} to ({axis_2}, {axis_1}) "
                "is not implemented"
            )


T = TypeVar("T", f32, f64)


@final
@dataclass(frozen=True, slots=True)
class Coordinates(Generic[F]):
    geometry: Geometry
    x1: FArray1D[F]
    x2: FArray1D[F]
    x3: FArray1D[F]

    def __post_init__(self) -> None:
        for attr in ("x1", "x2", "x3"):
            arr = getattr(self, attr)
            if len(arr) == 1:
                object.__setattr__(self, attr, np.repeat(arr, 2))

        x1 = self.x1
        x2 = self.x2
        x3 = self.x3
        excs = [
            (
                "multiple issues with input dtypes",
                collect_dtype_exceptions(x1.dtype, x2.dtype, x3.dtype),
            )
        ]
        if x1.ndim != 1 or x2.ndim != 1 or x3.ndim != 1:
            excs.append(
                (
                    "unused",
                    [
                        TypeError(
                            "Expected input arrays to all be 1D. "
                            f"Got {x1.ndim=}, {x2.ndim=}, {x3.ndim=}"
                        )
                    ],
                )
            )
        if exc := compile_exceptions("multiple issues with input arrays", *excs):
            raise exc

    def __eq__(self, other: Any) -> bool:
        # checks are ordered from cheapest to most expensive
        # so False is returned as early as possible when other
        # is not comparable
        return other is self or (
            type(other) is Coordinates
            and self.geometry == other.geometry
            and self.shape == other.shape
            and bool(
                (other.x3 is self.x3 or np.all(other.x3 == self.x3))
                and (other.x2 is self.x2 or np.all(other.x2 == self.x2))
                and (other.x1 is self.x1 or np.all(other.x1 == self.x1))
            )
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return len(self.x1), len(self.x2), len(self.x3)

    @property
    def dtype(self) -> np.dtype[F]:
        return self.x1.dtype

    def astype(self, dtype: np.dtype[T]) -> Coordinates[T]:
        """
        For convenience, mimic np.ndarray.astype

        The underlying data is always copied.

        .. versionadded: 0.20.0
        """
        return Coordinates(
            geometry=self.geometry,
            x1=self.x1.astype(dtype),
            x2=self.x2.astype(dtype),
            x3=self.x3.astype(dtype),
        )

    @property
    def axes(self) -> tuple[Axis, Axis, Axis]:
        return axes_from_geometry(self.geometry)

    def get_axis_index(self, axis: Axis) -> int:
        axes = list(self.axes)
        if axis not in axes:
            raise ValueError(f"axis {axis} isn't native to {self.geometry} geometry")
        return axes.index(axis)

    def get_axis_attr_name(self, axis: Axis) -> Literal["x1", "x2", "x3"]:
        match self.get_axis_index(axis):
            case 0:
                return "x1"
            case 1:
                return "x2"
            case 2:
                return "x3"
            case _ as _unreachable:
                raise AssertionError

    def get_axis_array(self, axis: Axis | str) -> FArray1D[F]:
        if isinstance(axis, str):
            axis = Axis.from_label(axis)
        idx = self.get_axis_index(axis)
        if idx == 0:
            arr = self.x1
        elif idx == 1:
            arr = self.x2
        elif idx == 2:
            arr = self.x3
        else:
            raise AssertionError
        return arr.copy()  # type: ignore[no-any-return]

    def get_axis_array_med(self, axis: Axis | str) -> FArray1D[F]:
        # convenience compatibility shim
        # should probably be removed or refactored
        if isinstance(axis, str):
            axis = Axis.from_label(axis)
        arr = self.get_axis_array(axis)
        return cast(FArray1D[F], 0.5 * (arr[1:] + arr[:-1]))

    def is_uniformly_spaced(self, axis: Axis) -> bool:
        base = self.get_axis_array(axis)
        dbase = np.diff(base)
        return bool(
            dbase.std() / dbase.max() / dbase.size < 5 * np.finfo(base.dtype).eps
        )

    # transformations
    def project_along(self, axis: Axis, position: float) -> Coordinates[F]:
        # TODO: stop using this method and deprecate it
        from nonos.api.tools import bracketing_values  # avoid an import cycle

        attr = self.get_axis_attr_name(axis)
        array = bracketing_values(
            self.get_axis_array(axis),
            position,
        ).as_array(dtype=self.dtype)
        return replace(self, **{attr: array})  # type: ignore

    def slice_at_index(self, axis: Axis, idx: int | AutoIndex, /) -> Coordinates[F]:
        attr = self.get_axis_attr_name(axis)
        arr = self.get_axis_array(axis)
        match idx:
            case AutoIndex.LEFTMOST:
                value = arr.min()
            case AutoIndex.RIGHTMOST:
                value = arr.max()
            case AutoIndex.MIDPOINT:
                value = 0.5 * (arr.max() + arr.min())
            case _:
                value = arr[idx]

        return replace(self, **{attr: np.array([value])})  # type: ignore

    def periodic_shift(self, axis: Axis, /, *, by: int) -> Coordinates[F]:
        # this represents the fundamental implementation of a rotation along
        # some curvilinear axis, but should accept any valid axis nonetheless:
        # the important part is that it should be able to assume periodicity, which
        # the required keyword argument 'period' hopefully conveys
        # Note that the axis specified is the direction of the shift in the
        # appropriate coordinates, *not* a normal, which would in general be ill-defined
        attr = self.get_axis_attr_name(axis)
        if not self.is_uniformly_spaced(axis):
            raise ValueError(
                "periodic shifting requires exactly equal "
                "grid spacing in the shift direction"
            )

        return replace(  # type: ignore
            self,
            **{attr: np.roll(self.get_axis_array(axis), shift=by)},  # type: ignore
        )

    def select_logical_slab(
        self,
        *,
        x1: IndexRange = SELECT_ALL,
        x2: IndexRange = SELECT_ALL,
        x3: IndexRange = SELECT_ALL,
    ) -> Coordinates[F]:
        return replace(
            self,
            x1=self.x1[x1.as_slice()],
            x2=self.x2[x2.as_slice()],
            x3=self.x3[x3.as_slice()],
        )

    def to_dict(self) -> StrDict:
        return {"geometry": self.geometry} | {
            axis.label: self.get_axis_array(axis) for axis in self.axes
        }

    @overload
    def native_from_wanted(self, axis_1: Axis, axis_2: None = None, /) -> Axis: ...
    @overload
    def native_from_wanted(
        self, axis_1: Axis, axis_2: Axis, /
    ) -> tuple[Axis, Axis]: ...

    def native_from_wanted(self, axis_1, axis_2=None, /):  # type: ignore[no-untyped-def]
        # compatibility shim for old Coordinates class
        # TODO: rename this
        if axis_2 is not None:
            return _native_plane_from_target_plane(self.geometry, axis_1, axis_2)
        else:
            return _native_axis_from_target_axis(self.geometry, axis_1)

    def _meshgrid_conversion_1d(self, axis_1: Axis) -> dict[Axis, FArray1D[F]]:
        native_plane_axis = self.native_from_wanted(axis_1)
        native_meshcoords = self._meshgrid_reduction(native_plane_axis, None)
        return native_meshcoords

    def _meshgrid_conversion_2d(
        self, axis_1: Axis, axis_2: Axis
    ) -> dict[Axis, FArray2D[F]]:
        native_plane_axes = self.native_from_wanted(axis_1, axis_2)
        native_meshcoords = self._meshgrid_reduction(*native_plane_axes)
        target_geometry = _get_target_geometry(axis_1, axis_2)
        return self.target_from_native(target_geometry, native_meshcoords)

    @overload
    def _meshgrid_reduction(
        self, axis_1: Axis, axis_2: None, /
    ) -> dict[Axis, FArray1D[F]]: ...
    @overload
    def _meshgrid_reduction(
        self, axis_1: Axis, axis_2: Axis, /
    ) -> dict[Axis, FArray2D[F]]: ...
    def _meshgrid_reduction(self, axis_1, axis_2, /):  # type: ignore[no-untyped-def]
        # TODO: this could easily be split into 2 functions (one for each dimensions)
        axes = axis_1, axis_2
        gaxes = axes_from_geometry(self.geometry)
        if axis_1 not in gaxes:
            raise ValueError(f"expected one of {gaxes}, got {axis_1}")
        if axis_2 is not None and axis_2 not in gaxes:
            raise ValueError(f"expected one of {gaxes}, got {axis_2}")

        if axis_2 is None:
            # 1D case
            return {axis_1: self.get_axis_array_med(axis_1)}
        else:
            # 2D case
            normal = set(gaxes).difference(axes).pop()
            a1, a2 = np.meshgrid(
                self.get_axis_array(axis_1), self.get_axis_array(axis_2)
            )
            return {
                axis_1: a1,
                axis_2: a2,
                # this is *not* consistent with the corresponding overload
                # typecheckers do not actually verify it
                # https://github.com/la-niche/nonos/issues/672
                normal: self.get_axis_array_med(normal),
            }

    def target_from_native(
        self,
        target_geometry: Geometry,
        coords: dict[Axis, FArray[D, F]],
    ) -> dict[Axis, FArray[D, F]]:
        match self.geometry:
            case Geometry.CARTESIAN:
                x = coords[Axis.CARTESIAN_X]
                y = coords[Axis.CARTESIAN_Y]
                z = coords[Axis.CARTESIAN_Z]
                match target_geometry:
                    case Geometry.CARTESIAN:
                        return {
                            Axis.CARTESIAN_X: x,
                            Axis.CARTESIAN_Y: y,
                            Axis.CARTESIAN_Z: z,
                        }
                    case Geometry.POLAR:
                        return {
                            Axis.CYLINDRICAL_RADIUS: np.sqrt(x**2 + y**2),
                            Axis.AZIMUTH: np.arctan2(y, x),
                            Axis.CARTESIAN_Z: z,
                        }
                    case Geometry.SPHERICAL:
                        return {
                            Axis.SPHERICAL_RADIUS: np.sqrt(x**2 + y**2 + z**2),
                            Axis.COLATITUDE: np.arctan2(np.sqrt(x**2 + y**2), z),
                            Axis.AZIMUTH: np.arctan2(y, x),
                        }
                    case _ as unreachable:
                        assert_never(unreachable)

            case Geometry.SPHERICAL:
                r = coords[Axis.SPHERICAL_RADIUS]
                theta = coords[Axis.COLATITUDE]
                phi = coords[Axis.AZIMUTH]
                match target_geometry:
                    case Geometry.CARTESIAN:
                        # note: I'm intentionally not reproducing a
                        # special case that I don't think is needed (theta.ndim<=1)
                        if theta.ndim <= 1:
                            # bug-for-bug compat
                            # this is extremely suspicious and should be inspected
                            # more thoroughly, I suspect it's just working around
                            # a completely different bug
                            return {
                                Axis.CARTESIAN_X: r * np.sin(theta) * np.cos(phi),
                                Axis.CARTESIAN_Y: r * np.sin(theta) * np.sin(phi),
                                Axis.CARTESIAN_Z: np.cos(theta),
                            }
                        return {
                            Axis.CARTESIAN_X: r * np.sin(theta) * np.cos(phi),
                            Axis.CARTESIAN_Y: r * np.sin(theta) * np.sin(phi),
                            Axis.CARTESIAN_Z: r * np.cos(theta),
                        }
                    case Geometry.POLAR:
                        return {
                            Axis.CYLINDRICAL_RADIUS: r * np.sin(theta),
                            Axis.AZIMUTH: phi,
                            Axis.CARTESIAN_Z: r * np.cos(theta),
                        }
                    case Geometry.SPHERICAL:
                        return {
                            Axis.SPHERICAL_RADIUS: r,
                            Axis.COLATITUDE: theta,
                            Axis.AZIMUTH: phi,
                        }
                    case _ as unreachable:
                        assert_never(unreachable)

            case Geometry.POLAR:
                R = coords[Axis.CYLINDRICAL_RADIUS]
                phi = coords[Axis.AZIMUTH]
                z = coords[Axis.CARTESIAN_Z]
                match target_geometry:
                    case Geometry.CARTESIAN:
                        return {
                            Axis.CARTESIAN_X: R * np.cos(phi),
                            Axis.CARTESIAN_Y: R * np.sin(phi),
                            Axis.CARTESIAN_Z: z,
                        }
                    case Geometry.POLAR:
                        return {
                            Axis.CYLINDRICAL_RADIUS: R,
                            Axis.AZIMUTH: phi,
                            Axis.CARTESIAN_Z: z,
                        }
                    case Geometry.SPHERICAL:
                        return {
                            Axis.SPHERICAL_RADIUS: np.sqrt(R**2 + z**2),
                            Axis.COLATITUDE: np.arctan2(R, z),
                            Axis.AZIMUTH: phi,
                        }
                    case _ as unreachable:
                        assert_never(unreachable)

            case _ as unreachable:
                assert_never(unreachable)

    def _safe_get_axis_array(self, attr: str) -> FArray1D[F]:
        try:
            return self.get_axis_array(attr)
        except KeyError:
            raise AttributeError(
                f"Coordinates object has no attribute {attr}"
            ) from None

    def _safe_get_axis_array_med(self, attr: str) -> FArray1D[F]:
        try:
            return self.get_axis_array_med(attr)
        except KeyError:
            raise AttributeError(
                f"Coordinates object has no attribute {attr}"
            ) from None

    @property
    @deprecated(
        "Coordinates.x is deprecated since v0.18.0 "
        "and may be removed in a future version. "
        "Instead, use Coordinates.get_axis_array('x')"
    )
    def x(self) -> FArray1D[F]:
        return self._safe_get_axis_array("x")

    @property
    @deprecated(
        "Coordinates.y is deprecated since v0.18.0 "
        "and may be removed in a future version. "
        "Instead, use Coordinates.get_axis_array('y')"
    )
    def y(self) -> FArray1D[F]:
        return self._safe_get_axis_array("y")

    @property
    @deprecated(
        "Coordinates.z is deprecated since v0.18.0 "
        "and may be removed in a future version. "
        "Instead, use Coordinates.get_axis_array('z')"
    )
    def z(self) -> FArray1D[F]:
        return self._safe_get_axis_array("z")

    @property
    @deprecated(
        "Coordinates.r is deprecated since v0.18.0 "
        "and may be removed in a future version. "
        "Instead, use Coordinates.get_axis_array('r')"
    )
    def r(self) -> FArray1D[F]:
        return self._safe_get_axis_array("r")

    @property
    @deprecated(
        "Coordinates.theta is deprecated since v0.18.0 "
        "and may be removed in a future version. "
        "Instead, use Coordinates.get_axis_array('theta')"
    )
    def theta(self) -> FArray1D[F]:
        return self._safe_get_axis_array("theta")

    @property
    @deprecated(
        "Coordinates.phi is deprecated since v0.18.0 "
        "and may be removed in a future version. "
        "Instead, use Coordinates.get_axis_array('phi')"
    )
    def phi(self) -> FArray1D[F]:
        return self._safe_get_axis_array("phi")

    @property
    @deprecated(
        "Coordinates.R is deprecated since v0.18.0 "
        "and may be removed in a future version. "
        "Instead, use Coordinates.get_axis_array('R')"
    )
    def R(self) -> FArray1D[F]:
        return self._safe_get_axis_array("R")

    @property
    @deprecated(
        "Coordinates.xmed is deprecated since v0.18.0 "
        "and may be removed in a future version. "
        "Instead, use Coordinates.get_axis_array_med('x')"
    )
    def xmed(self) -> FArray1D[F]:
        return self._safe_get_axis_array_med("x")

    @property
    @deprecated(
        "Coordinates.ymed is deprecated since v0.18.0 "
        "and may be removed in a future version. "
        "Instead, use Coordinates.get_axis_array_med('y')"
    )
    def ymed(self) -> FArray1D[F]:
        return self._safe_get_axis_array_med("y")

    @property
    @deprecated(
        "Coordinates.zmed is deprecated since v0.18.0 "
        "and may be removed in a future version. "
        "Instead, use Coordinates.get_axis_array_med('z')"
    )
    def zmed(self) -> FArray1D[F]:
        return self._safe_get_axis_array_med("z")

    @property
    @deprecated(
        "Coordinates.rmed is deprecated since v0.18.0 "
        "and may be removed in a future version. "
        "Instead, use Coordinates.get_axis_array_med('r')"
    )
    def rmed(self) -> FArray1D[F]:
        return self._safe_get_axis_array_med("r")

    @property
    @deprecated(
        "Coordinates.thetamed is deprecated since v0.18.0 "
        "and may be removed in a future version. "
        "Instead, use Coordinates.get_axis_array_med('theta')"
    )
    def thetamed(self) -> FArray1D[F]:
        return self._safe_get_axis_array_med("theta")

    @property
    @deprecated(
        "Coordinates.phimed is deprecated since v0.18.0 "
        "and may be removed in a future version. "
        "Instead, use Coordinates.get_axis_array_med('phi')"
    )
    def phimed(self) -> FArray1D[F]:
        return self._safe_get_axis_array_med("phi")

    @property
    @deprecated(
        "Coordinates.Rmed is deprecated since v0.18.0 "
        "and may be removed in a future version. "
        "Instead, use Coordinates.get_axis_array_med('R')"
    )
    def Rmed(self) -> FArray1D[F]:
        return self._safe_get_axis_array_med("R")
