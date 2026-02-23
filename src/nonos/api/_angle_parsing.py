import sys
from typing import Protocol

if sys.version_info >= (3, 13):
    from warnings import deprecated
else:
    from typing_extensions import deprecated


class PlanetAzimuthFinder(Protocol):
    def __call__(self, *, planet_file: str) -> float: ...


def _resolve_planet_file(
    *,
    planet_file: str | None = None,
    planet_number: int | None = None,
) -> str:
    if planet_number is not None and planet_file is not None:
        raise TypeError(
            "received both planet_number and planet_file arguments. "
            "Please pass at most one."
        )
    if planet_file is not None:
        return planet_file
    else:
        return f"planet{planet_number or 0}.dat"


@deprecated(
    "_parse_planet_file is deprecated since v0.20.0 "
    "and may be removed in a future version. "
    "Use _resolve_planet_file instead"
)
def _parse_planet_file(
    *,
    planet_file: str | None = None,
    planet_number: int | None = None,
) -> str:  # pragma: no cover
    # backward compatibility layer for nonos-cli 0.1.0
    return _resolve_planet_file(planet_file=planet_file, planet_number=planet_number)


def _resolve_rotate_by(
    *,
    rotate_by: float | None,
    rotate_with: str | None,
    planet_azimuth_finder: PlanetAzimuthFinder,
) -> float:
    if not (defined_args := {rotate_with, rotate_by} - {None}):
        # no rotation specified
        return 0.0

    if len(defined_args) > 1:
        raise TypeError(
            "rotate_by and rotate_with cannot be specified at the same time"
        )

    # beyond this point, we know that exactly one parameter was specified,
    # let's funnel it down to a rotate_by form
    if rotate_with is not None:
        rotate_by = planet_azimuth_finder(planet_file=rotate_with)

    if rotate_by is None:
        # this is never supposed to happen, but it's needed to convince mypy that
        # we will not return a None
        raise AssertionError

    return rotate_by
