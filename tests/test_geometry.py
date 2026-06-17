import sys
from itertools import chain, combinations, permutations

import numpy as np
import numpy.testing as npt
import pytest
from pytest import RaisesExc, RaisesGroup

from nonos._geometry import (
    _AXIS_TO_STR,
    _STR_TO_AXIS,
    AutoIndex,
    Coordinates,
    IndexRange,
    _get_target_geometry,
    _native_axis_from_target_axis,
    _native_plane_from_target_plane,
)
from nonos.geometry import Axis, Geometry

if sys.version_info >= (3, 13):
    from copy import replace
else:
    from dataclasses import replace


@pytest.mark.parametrize(
    "axes, expected_geometry",
    list(
        chain.from_iterable(
            [[(axes, g) for axes in combinations(g.axes, 2)] for g in Geometry]
        )
    ),
)
def test_get_target_geometry(axes, expected_geometry):
    ax1, ax2 = axes
    assert _get_target_geometry(ax1, ax2) is expected_geometry
    assert _get_target_geometry(ax2, ax1) is expected_geometry


@pytest.mark.parametrize("axis", set(chain.from_iterable(g.axes for g in Geometry)))
@pytest.mark.parametrize("native_geometry", Geometry)
def test_native_axis_from_target_axis(native_geometry, axis):
    try:
        axo = _native_axis_from_target_axis(native_geometry, axis)
    except NotImplementedError:
        pytest.skip("not a real issue, but let's make it visible")
    else:
        assert axo in native_geometry.axes


@pytest.mark.parametrize(
    "axes",
    list(chain.from_iterable(combinations(g.axes, 2) for g in Geometry)),
)
@pytest.mark.parametrize("native_geometry", Geometry)
def test_native_plane_from_target_plane(native_geometry, axes):
    axi1, axi2 = axes
    try:
        axo1, axo2 = _native_plane_from_target_plane(native_geometry, axi1, axi2)
    except NotImplementedError:
        pytest.skip("not a real issue, but let's make it visible")
    else:
        assert axo1 != axi2
        assert {axo1, axo2}.issubset(native_geometry.axes)

        # also check that swapping inputs results in swapped outputs
        res2 = _native_plane_from_target_plane(native_geometry, axi2, axi1)
        assert res2 == (axo2, axo1)


def test_module_constants():
    assert set(_STR_TO_AXIS.values()) == set(Axis)
    assert set(_AXIS_TO_STR.keys()) == set(Axis)


@pytest.mark.parametrize(
    "dt, ctx",
    [
        pytest.param(
            "f4",
            RaisesExc(TypeError, match=r"^mixed dtype itemsizes: \[4, 8\]$"),
            id="itemsize",
        ),
        pytest.param(
            "i8",
            RaisesExc(TypeError, match=r"^mixed dtype kinds: \['f', 'i'\]$"),
            id="kind",
        ),
        pytest.param(
            "i4",
            RaisesGroup(
                RaisesExc(TypeError, match=r"^mixed dtype itemsizes: \[4, 8\]$"),
                RaisesExc(TypeError, match=r"^mixed dtype kinds: \['f', 'i'\]$"),
                match="multiple issues with input dtypes",
            ),
            id="all",
        ),
    ],
)
def test_coordinates_mismatch_dtypes(dt, ctx):
    with ctx:
        Coordinates(
            geometry=Geometry.CARTESIAN,
            x1=np.linspace(0, 5, 8, dtype=">f8"),
            x2=np.linspace(0, 5, 8, dtype=">f8"),
            x3=np.arange(5, dtype=dt),
        )


@pytest.mark.parametrize(
    "x1, x2, x3",
    list(
        permutations(
            [
                np.linspace(0, 8, 8),
                np.linspace(0, 8, 8),
                np.linspace(0, 8, 8).reshape(4, 2),
            ]
        )
    ),
)
def test_coordinates_invalid_ndim(x1, x2, x3):
    with pytest.raises(
        TypeError,
        match=(
            "^Expected input arrays to all be 1D. "
            f"Got {x1.ndim=}, {x2.ndim=}, {x3.ndim=}$"
        ),
    ):
        Coordinates(geometry=Geometry.CARTESIAN, x1=x1, x2=x2, x3=x3)


@pytest.mark.parametrize(
    "geometry",
    [
        Geometry.CARTESIAN,
        Geometry.POLAR,
        Geometry.SPHERICAL,
    ],
)
def test_deprecated_coordinates_array_accesses(geometry):
    x = np.linspace(0, 1, 5)  # should be valid in any axis
    coords = Coordinates(geometry, x, x, x)
    for axis in geometry.axes:
        attr = axis.label
        with pytest.deprecated_call():
            arr = getattr(coords, attr)
        npt.assert_array_equal(arr, coords.get_axis_array(attr))

        with pytest.deprecated_call():
            arrmed = getattr(coords, f"{attr}med")
        npt.assert_array_equal(arrmed, coords.get_axis_array_med(attr))


def test_slice_at_index(subtests):
    x1 = np.linspace(-1, 1, 10)
    x2 = np.linspace(-2, 2, 20)
    x3 = np.linspace(-3_000, 4_000, 300_000)
    c0 = Coordinates(Geometry.CARTESIAN, x1, x2, x3)

    with subtests.test("x"):
        c1 = c0.slice_at_index(Axis.CARTESIAN_X, 5)
        assert c1.shape == (2, x2.size, x3.size)
        assert np.all(c1.x1 == c0.x1[5])

        c1L = c0.slice_at_index(Axis.CARTESIAN_X, AutoIndex.LEFTMOST)
        assert c1L.shape == c1.shape
        assert np.all(c1L.x1 == c0.x1[0])

    with subtests.test("y"):
        c2 = c0.slice_at_index(Axis.CARTESIAN_Y, 8)
        assert c2.shape == (x1.size, 2, x3.size)
        assert np.all(c2.x2 == c0.x2[8])

        c2R = c0.slice_at_index(Axis.CARTESIAN_Y, AutoIndex.RIGHTMOST)
        assert c2R.shape == c2.shape
        assert np.all(c2R.x2 == c0.x2[-1])

    with subtests.test("z"):
        c3 = c0.slice_at_index(Axis.CARTESIAN_Z, -100)
        assert c3.shape == (x1.size, x2.size, 2)
        assert np.all(c3.x3 == c0.x3[-100])

        c3M = c0.slice_at_index(Axis.CARTESIAN_Z, AutoIndex.MIDPOINT)
        assert c3M.shape == c3.shape
        # not that 500 itself isn't part of c0.c3
        assert np.all(c3M.x3 == 500.0)


def test_periodic_shift(subtests):
    x1 = np.linspace(-1, 1, 10)
    x2 = np.linspace(-2, 2, 20)
    x3 = np.linspace(-3_000, 4_000, 300_000)
    c0 = Coordinates(Geometry.CARTESIAN, x1, x2, x3)

    with subtests.test("x"):
        c1 = c0.periodic_shift(Axis.CARTESIAN_X, by=5)
        assert c1 == replace(c0, x1=np.roll(x1, shift=5))

    with subtests.test("y"):
        c2 = c0.periodic_shift(Axis.CARTESIAN_Y, by=10)
        assert c2 == replace(c0, x2=np.roll(x2, shift=10))

    with subtests.test("z"):
        c3 = c0.periodic_shift(Axis.CARTESIAN_Z, by=20)
        assert c3 == replace(c0, x3=np.roll(x3, shift=20))


def test_index_range_as_slice():
    ir = IndexRange(1, 5)
    assert ir.as_slice() == slice(1, 5)


@pytest.mark.parametrize(
    "kwargs, expected_msg",
    [
        ({"start": -1}, "Expected start>=0, got start=-1"),
        ({"start": -1, "stop": 2}, "Expected start>=0, got start=-1"),
        ({"start": 1, "stop": 0}, "Expected stop>0, got stop=0"),
        ({"stop": 0}, "Expected stop>0, got stop=0"),
        ({"start": 5, "stop": 4}, "Expected start<stop, got start=5, stop=4"),
    ],
)
def test_index_range_invalid_input(kwargs, expected_msg):
    with pytest.raises(ValueError, match=expected_msg):
        IndexRange(**kwargs)


def test_logical_slab(subtests):
    x1 = np.linspace(-1, 1, 10)
    x2 = np.linspace(-2, 2, 20)
    x3 = np.linspace(-3_000, 4_000, 300_000)
    c0 = Coordinates(Geometry.CARTESIAN, x1, x2, x3)

    with subtests.test("x1"):
        c1 = c0.select_logical_slab(x1=IndexRange(stop=8))
        assert c1 == replace(c0, x1=x1[:8])

    with subtests.test("x1, x2"):
        c2 = c0.select_logical_slab(
            x1=IndexRange(stop=8),
            x2=IndexRange(start=18, stop=19),
        )
        assert c2 == replace(c0, x1=x1[:8], x2=np.full(2, x2[18], dtype=x2.dtype))

    with subtests.test("x2, x3"):
        c3 = c0.select_logical_slab(
            x2=IndexRange(start=18, stop=19),
            x3=IndexRange(start=1000, stop=1010),
        )
        assert c3 == replace(
            c0, x2=np.full(2, x2[18], dtype=x2.dtype), x3=x3[1000:1010]
        )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_periodic_shift_stretched_axis(dtype):
    c0 = Coordinates(
        Geometry.CARTESIAN,
        x1=np.linspace(-1, 1, 10),
        x2=np.linspace(-2, 2, 20),
        x3=np.geomspace(1, 4, 30),
    ).astype(dtype)

    with pytest.raises(
        ValueError,
        match=(
            r"^periodic shifting requires exactly equal "
            r"grid spacing in the shift direction$"
        ),
    ):
        c0.periodic_shift(Axis.CARTESIAN_Z, by=20)
