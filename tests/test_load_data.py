import os

import pytest

from nonos.api import GasDataSet
from nonos.api.from_simulation import _load_fargo3d, _load_fargo_adsg, _load_idefix


@pytest.mark.parametrize(
    ("func", "on", "subdir", "kwargs", "expected_geometry", "expected_shape"),
    [
        (
            _load_idefix,
            500,
            "idefix_spherical_planet3d",
            {"cell": "edges", "computedata": True},
            "spherical",
            (72, 32, 196),
        ),
        (_load_fargo3d, 40, "fargo3d_planet2d", {}, "polar", (256, 256, 1)),
        (_load_fargo_adsg, 100, "fargo_adsg", {}, "polar", (612, 900, 1)),
    ],
)
def test_load(
    test_data_dir, func, on, subdir, kwargs, expected_geometry, expected_shape
):
    ds = func(
        on,
        directory=test_data_dir / subdir,
        **kwargs,
    )

    assert ds.geometry == expected_geometry
    assert ds.grid.shape == expected_shape


def test_api_vtk_by_name(test_data_dir):
    os.chdir(test_data_dir / "idefix_spherical_planet3d")

    GasDataSet(500, pattern=lambda on: f"data.{on:04d}.vtk")

    with pytest.raises(
        FileNotFoundError, match="Idefix: datawrong.0500.vtk not found."
    ):
        GasDataSet(500, pattern=lambda on: f"datawrong.{on:04d}.vtk")
