import os
import shutil

import numpy as np
import pytest

from nonos.api import GasDataSet


def test_from_npy_error(test_data_dir):
    os.chdir(test_data_dir / "idefix_spherical_planet3d")

    with pytest.raises(
        FileNotFoundError, match="Original output was not reduced, or file"
    ):
        GasDataSet.from_npy(500, operation="azimuthal_averag")


def test_roundtrip_simple(test_data_dir, tmp_path):
    os.chdir(test_data_dir / "idefix_spherical_planet3d")

    ds = GasDataSet(500)
    assert ds.nfields == 7

    gf = ds["RHO"].azimuthal_average()

    gf.save(tmp_path)
    dsnpy = GasDataSet.from_npy(500, operation="azimuthal_average", directory=tmp_path)
    assert dsnpy.nfields == 1


def test_roundtrip_simple_pluto(test_data_dir):
    os.chdir(test_data_dir / "pluto_spherical")

    ds = GasDataSet.from_npy(7283, operation="azimuthal_average")
    assert ds.nfields == 1


def test_roundtrip_no_operation_all_field(test_data_dir, tmp_path):
    os.chdir(test_data_dir / "idefix_spherical_planet3d")

    ds = GasDataSet(500)
    assert ds.nfields == 7

    gf = ds["RHO"]

    gf.save(tmp_path)
    dsnpy = GasDataSet.from_npy(500, operation="", directory=tmp_path)
    assert dsnpy.nfields == 1
    np.testing.assert_array_equal(ds["RHO"].data, dsnpy["RHO"].data)


def test_roundtrip_other_dir(test_data_dir, tmp_path):
    os.chdir(test_data_dir / "idefix_spherical_planet3d")
    gf = GasDataSet(500)["RHO"].azimuthal_average()
    gf.save(tmp_path)
    dsnpy = GasDataSet.from_npy(500, operation="azimuthal_average", directory=tmp_path)
    assert dsnpy.nfields == 1


def test_roundtrip_current_dir(test_data_dir):
    os.chdir(test_data_dir / "idefix_spherical_planet3d")
    gf = GasDataSet(500)["RHO"].azimuthal_average()
    gf.save()
    dsnpy = GasDataSet.from_npy(500, operation="azimuthal_average")
    assert dsnpy.nfields == 1
    shutil.rmtree("rho")
    shutil.rmtree("header")

@pytest.mark.parametrize(
    "from_abs_path",
    [
        pytest.param(True, id="from absolute path"),
        pytest.param(False, id="from relative path"),
    ],
)
def test_api_vtk_by_name(test_data_dir, from_abs_path):
    data_dir = test_data_dir / "idefix_spherical_planet3d"
    if from_abs_path:
        input_ = str((data_dir / "data.0500.vtk").absolute())
    else:
        os.chdir(data_dir)
        input_ = "data.0500.vtk"

    ds = GasDataSet(input_)
    assert ds.on == 500

    with pytest.raises(
        FileNotFoundError, match=r"In idfxReadVTK: .*datawrong\.0500\.vtk not found\."
    ):
        GasDataSet(input_.replace("data.0500", "datawrong.0500"))


def test_api_vtk_by_name_fargo(test_data_dir):
    os.chdir(test_data_dir / "fargo3d_planet2d")

    with pytest.raises(TypeError, match="on can only be an int for fargo3d"):
        GasDataSet(f"gasdens{40:04d}.dat")


def test_api_fluid_fargo3d(test_data_dir):
    os.chdir(test_data_dir / "fargo3d_multifluid")

    on = 5

    ds = GasDataSet(on, fluid="dust2")
    assert ds.nfields == 1

    with pytest.raises(
        FileNotFoundError,
        match=r"No file matches the pattern 'dust4\*5\.dat'",
    ):
        GasDataSet(on, fluid="dust4")


def test_api_fluid_idefix(test_data_dir):
    os.chdir(test_data_dir / "idefix_spherical_planet3d")

    on = 500

    with pytest.raises(ValueError, match="fluid is defined only for fargo3d outputs"):
        GasDataSet(on, fluid="dust1")
