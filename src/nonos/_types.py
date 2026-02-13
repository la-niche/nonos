__all__ = [
    "D1",
    "D2",
    "D3",
    "BinData",
    "BinReader",
    "D",
    "F",
    "FArray",
    "FArray1D",
    "FArray2D",
    "FArray3D",
    "FloatArray",  # to be replaced by FArray
    "FrameType",
    "IniData",
    "IniReader",
    "OrbitalElements",
    "PlanetData",
    "PlanetReader",
    "StrDict",
]
import os
import sys
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeAlias, TypeVar, final

import numpy as np
from numpy import float32 as f32, float64 as f64

if sys.version_info >= (3, 11):
    from typing import Self, assert_never
else:
    from typing_extensions import Self, assert_never

if TYPE_CHECKING:
    from nonos._geometry import Geometry

StrDict: TypeAlias = dict[str, Any]

D1: TypeAlias = tuple[int]
D2: TypeAlias = tuple[int, int]
D3: TypeAlias = tuple[int, int, int]

D = TypeVar("D", D1, D2, D3)
F = TypeVar("F", f32, f64)
FloatArray: TypeAlias = np.ndarray[Any, np.dtype[F]]

FArray: TypeAlias = np.ndarray[D, np.dtype[F]]
FArray1D: TypeAlias = FArray[D1, F]
FArray2D: TypeAlias = FArray[D2, F]
FArray3D: TypeAlias = FArray[D3, F]


class FrameType(Enum):
    FIXED_FRAME = auto()
    CONSTANT_ROTATION = auto()
    PLANET_COROTATION = auto()


@final
@dataclass(frozen=True, eq=False, slots=True)
class BinData(Generic[F]):
    data: StrDict
    geometry: "Geometry"
    x1: FArray1D[F]
    x2: FArray1D[F]
    x3: FArray1D[F]

    @classmethod
    def default_init(cls) -> "BinData[F]":
        return BinData(
            **(
                {  # type: ignore
                    field.name: field.default
                    for field in cls.__dataclass_fields__.values()
                }
                | {"data": {}}
            )
        )

    def finalize(self) -> Self:
        missing_fields = [
            f.name
            for f in self.__dataclass_fields__.values()
            if getattr(self, f.name) is f.default
        ]
        if any(missing_fields):
            raise TypeError(
                f"The following fields were not initialized: {missing_fields}"
            )
        return self


@final
@dataclass(frozen=True, eq=False, slots=True)
class OrbitalElements(Generic[F]):
    i: FArray1D[F]
    e: FArray1D[F]
    a: FArray1D[F]


@final
@dataclass(frozen=True, eq=False)
class PlanetData(Generic[F]):
    # fields that are required at __init__
    _init_attrs = ["x", "y", "z", "vx", "vy", "vz", "q", "t", "dt"]
    # additional derived field that can be computed on the fly
    _post_init_attrs = ["d"]
    __slots__ = _init_attrs + _post_init_attrs

    # cartesian position
    x: FArray1D[F]
    y: FArray1D[F]
    z: FArray1D[F]

    # cartesian velocity
    vx: FArray1D[F]
    vy: FArray1D[F]
    vz: FArray1D[F]

    # mass ratio (or mass in units of the central star's)
    q: FArray1D[F]

    # time and timestep
    t: FArray1D[F]
    dt: FArray1D[F]

    def __post_init__(self) -> None:
        object.__setattr__(self, "d", np.sqrt(self.x**2 + self.y**2 + self.z**2))

    def get_orbital_elements(self, frame: FrameType) -> OrbitalElements[F]:
        match frame:
            case FrameType.FIXED_FRAME:
                hx = self.y * self.vz - self.z * self.vy
                hy = self.z * self.vx - self.x * self.vz
                hz = self.x * self.vy - self.y * self.vx
                hhor = np.hypot(hx, hy)

                h2 = hx * hx + hy * hy + hz * hz
                h = np.sqrt(h2)
                i = np.arcsin(hhor / h)

                d = object.__getattribute__(self, "d")
                Ax = self.vy * hz - self.vz * hy - (1.0 + self.q) * self.x / d
                Ay = self.vz * hx - self.vx * hz - (1.0 + self.q) * self.y / d
                Az = self.vx * hy - self.vy * hx - (1.0 + self.q) * self.z / d

                e = np.sqrt(Ax * Ax + Ay * Ay + Az * Az) / (1.0 + self.q)
                a = h * h / ((1.0 + self.q) * (1.0 - e * e))
                return OrbitalElements(i, e, a)
            case FrameType.CONSTANT_ROTATION:
                raise NotImplementedError(
                    f"PlanetData.set_orbital_elements isn't implemented for {frame=}"
                )
            case FrameType.PLANET_COROTATION:
                # bug-for-bug compat
                return self.get_orbital_elements(FrameType.FIXED_FRAME)
            case _ as unreachable:
                assert_never(unreachable)

    def get_rotational_rate(self) -> FArray1D[F]:
        d = self.d  # type: ignore [attr-defined]
        return np.sqrt((1.0 + self.q) / pow(d, 3.0))  # type: ignore[no-any-return]


for key in PlanetData._post_init_attrs:
    PlanetData.__annotations__[key] = FArray1D


@final
@dataclass(frozen=True, slots=True)
class IniData:
    file: Path
    frame: FrameType
    rotational_rate: float
    output_time_interval: float
    meta: StrDict


class BinReader(Protocol, Generic[F]):
    @staticmethod
    def parse_output_number_and_filename(
        file_or_number: os.PathLike[str] | int,
        *,
        directory: os.PathLike[str],
        prefix: str,
    ) -> tuple[int, Path]: ...

    @staticmethod
    def get_bin_files(directory: os.PathLike[str], /) -> list[Path]: ...

    @staticmethod
    def read(file: os.PathLike[str], /, **meta: Any) -> BinData[F]: ...


class PlanetReader(Protocol, Generic[F]):
    @staticmethod
    def get_planet_files(directory: Path, /) -> list[Path]: ...

    @staticmethod
    def read(file: os.PathLike[str], /) -> PlanetData[F]: ...


class IniReader(Protocol):
    @staticmethod
    def read(file: os.PathLike[str], /) -> IniData: ...
