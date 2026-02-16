from itertools import chain, combinations

import numpy as np
import numpy.testing as npt
import pytest

from nonos._geometry import (
    _AXIS_TO_STR,
    _STR_TO_AXIS,
    Axis,
    Coordinates,
    Geometry,
    _get_target_geometry,
    _native_axis_from_target_axis,
    _native_plane_from_target_plane,
    axes_from_geometry,
)


@pytest.mark.parametrize(
    "axes, expected_geometry",
    list(
        chain.from_iterable(
            [
                [(axes, g) for axes in combinations(axes_from_geometry(g), 2)]
                for g in Geometry
            ]
        )
    ),
)
def test_get_target_geometry(axes, expected_geometry):
    ax1, ax2 = axes
    assert _get_target_geometry(ax1, ax2) is expected_geometry
    assert _get_target_geometry(ax2, ax1) is expected_geometry


@pytest.mark.parametrize(
    "axis",
    set(chain.from_iterable(axes_from_geometry(g) for g in Geometry)),
)
@pytest.mark.parametrize("native_geometry", Geometry)
def test_native_axis_from_target_axis(native_geometry, axis):
    try:
        axo = _native_axis_from_target_axis(native_geometry, axis)
    except NotImplementedError:
        pytest.skip("not a real issue, but let's make it visible")
    else:
        assert axo in axes_from_geometry(native_geometry)


@pytest.mark.parametrize(
    "axes",
    list(chain.from_iterable(combinations(axes_from_geometry(g), 2) for g in Geometry)),
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
        assert {axo1, axo2}.issubset(axes_from_geometry(native_geometry))

        # also check that swapping inputs results in swapped outputs
        res2 = _native_plane_from_target_plane(native_geometry, axi2, axi1)
        assert res2 == (axo2, axo1)


def test_module_constants():
    assert set(_STR_TO_AXIS.values()) == set(Axis)
    assert set(_AXIS_TO_STR.keys()) == set(Axis)


@pytest.mark.parametrize(
    "geometry",
    [Geometry.POLAR, Geometry.SPHERICAL],
)
def test_deprecated_coordinates_array_accesses(geometry):
    axes = axes_from_geometry(geometry)
    x = np.linspace(0, 1, 5)  # should be valid in any axis
    coords = Coordinates(geometry, x, x, x)
    for axis in axes:
        attr = axis.label
        with pytest.deprecated_call():
            arr = getattr(coords, attr)
        npt.assert_array_equal(arr, coords.get_axis_array(attr))

        with pytest.deprecated_call():
            arrmed = getattr(coords, f"{attr}med")
        npt.assert_array_equal(arrmed, coords.get_axis_array_med(attr))
