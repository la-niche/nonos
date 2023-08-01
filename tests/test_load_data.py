import os

import pytest

from nonos.api import GasDataSet


def test_api_vtk_by_name(test_data_dir):
    os.chdir(test_data_dir / "idefix_spherical_planet3d")

    GasDataSet(500, pattern=lambda on: f"data.{on:04d}.vtk")

    with pytest.raises(
        FileNotFoundError, match="In idfxReadVTK: datawrong.0500.vtk not found."
    ):
        GasDataSet(500, pattern=lambda on: f"datawrong.{on:04d}.vtk")
