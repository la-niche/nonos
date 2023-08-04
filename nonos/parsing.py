from typing import Any, List, Optional, Tuple, Union, overload

import numpy as np


def is_set(x: Any) -> bool:
    return x not in (None, "unset")


def parse_output_number_range(
    on: Optional[Union[List[int], int, str]], maxval: Optional[int] = None
) -> List[int]:
    if not is_set(on):
        if maxval is None:
            raise ValueError("Can't parse a range from unset values without a max.")
        return [maxval]

    if isinstance(on, int):
        return [on]

    assert isinstance(on, list) and all(isinstance(o, int) for o in on)

    if len(on) > 3:
        raise ValueError(
            f"Can't parse a range from sequence {on} with more than 3 values."
        )
    if len(on) == 1:
        return on

    if on[1] < on[0]:
        raise ValueError("Can't parse a range with max < min.")

    # make the upper boundary inclusive
    on[1] += 1
    ret = list(range(*on))
    if maxval is not None and (max_requested := ret[-1]) > maxval:
        raise ValueError(
            f"No output beyond {maxval} is available, but {max_requested} was requested."
        )
    return ret


def parse_range(extent, dim: int) -> Tuple[Optional[float], ...]:
    if not is_set(extent):
        if dim == 2:
            return (None, None, None, None)
        elif dim == 1:
            return (None, None)
        else:
            raise ValueError("dim has to be 1 or 2.")

    if len(extent) != 2 * dim:
        raise ValueError(
            f"Received sequence `extent` with incorrect size {len(extent)}. Expected exactly {2*dim=} values."
        )
    return tuple(float(i) if i != "x" else None for i in extent)


@overload
def range_converter(
    extent: Tuple[Optional[float], Optional[float]],
    abscissa: np.ndarray,
    ordinate: np.ndarray,
) -> Tuple[float, float]:
    ...


@overload
def range_converter(
    extent: Tuple[Optional[float], Optional[float], Optional[float], Optional[float]],
    abscissa: np.ndarray,
    ordinate: np.ndarray,
) -> Tuple[float, float, float, float]:
    ...


def range_converter(extent, abscissa, ordinate):
    if len(extent) == 4:
        return (
            _ if (_ := extent[0]) is not None else abscissa.min(),
            _ if (_ := extent[1]) is not None else abscissa.max(),
            _ if (_ := extent[2]) is not None else ordinate.min(),
            _ if (_ := extent[3]) is not None else ordinate.max(),
        )
    elif len(extent) == 2:
        return (
            _ if (_ := extent[0]) is not None else abscissa.min(),
            _ if (_ := extent[1]) is not None else abscissa.max(),
        )
    else:
        raise TypeError(f"Expected extent to be of lenght 2 or 4, got {len(extent)=}")


def parse_image_format(s: Optional[str]) -> str:
    from matplotlib.backend_bases import FigureCanvasBase

    if not is_set(s):
        return FigureCanvasBase.get_default_filetype()

    assert isinstance(s, str)
    _, _, ext = s.rpartition(".")
    if ext not in (
        available := list(FigureCanvasBase.get_supported_filetypes().keys())
    ):
        raise ValueError(
            f"Received unknown file format '{s}'. "
            f"Available formated are {available}."
        )
    return ext
