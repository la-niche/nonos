import os
import shutil

import numpy as np
import pytest

from nonos.api import GasDataSet


def test_from_npy_error(test_data_dir):
    with pytest.raises(FileNotFoundError):
        GasDataSet(
            500,
            operation="typo",
            directory=test_data_dir / "idefix_spherical_planet3d",
        )


def test_roundtrip_simple(test_data_dir, tmp_path):
    ds = GasDataSet(500, directory=test_data_dir / "idefix_spherical_planet3d")
    assert ds.nfields == 7

    gf = ds["RHO"].azimuthal_average()

    gf.save(tmp_path)
    dsnpy = GasDataSet(
        500,
        operation="azimuthal_average",
        directory=tmp_path,
    )
    assert dsnpy.nfields == 1


@pytest.mark.parametrize(
    "implicit_directory",
    [
        pytest.param(True, id="implicit directory"),
        pytest.param(False, id="explicit directory"),
    ],
)
def test_simple_fargo_adsg(test_data_dir, implicit_directory):
    data_dir = test_data_dir / "fargo_adsg_planet"
    if implicit_directory:
        os.chdir(data_dir)
        directory = None
    else:
        directory = data_dir.absolute()

    ds = GasDataSet(
        200,
        code="fargo_adsg",
        inifile="planetpendragon_200k.par",
        directory=directory,
    )
    assert ds.nfields == 1


def test_roundtrip_no_operation_all_field(test_data_dir, tmp_path):
    os.chdir(test_data_dir / "idefix_spherical_planet3d")

    ds = GasDataSet(500)
    assert ds.nfields == 7

    gf = ds["RHO"]

    gf.save(tmp_path)
    dsnpy = GasDataSet(
        500,
        operation="",
        directory=tmp_path,
    )
    assert dsnpy.nfields == 1
    np.testing.assert_array_equal(ds["RHO"].data, dsnpy["RHO"].data)


def test_roundtrip_other_dir(test_data_dir, tmp_path):
    os.chdir(test_data_dir / "idefix_spherical_planet3d")
    gf = GasDataSet(500)["RHO"].azimuthal_average()
    gf.save(tmp_path)
    dsnpy = GasDataSet(
        500,
        operation="azimuthal_average",
        directory=tmp_path,
    )
    assert dsnpy.nfields == 1


def test_npy_radial_at_r(test_data_dir, tmp_path):
    os.chdir(test_data_dir / "idefix_spherical_planet3d")
    gf = GasDataSet(500)["RHO"].radial_at_r(1.1)
    gf.save(tmp_path)
    dsnpy = GasDataSet(
        500,
        operation="radial_at_r1.1",
        directory=tmp_path,
    )
    assert list(dsnpy.keys()) == ["RHO"]


def test_save_current_dir(test_data_dir, tmp_path):
    os.chdir(test_data_dir / "idefix_spherical_planet3d")
    shutil.copy("idefix.ini", tmp_path / "idefix.ini")
    shutil.copy("data.0500.vtk", tmp_path / "data.0500.vtk")
    os.chdir(tmp_path)
    time_inifile = os.path.getmtime("idefix.ini")
    gf = GasDataSet(500)["RHO"]
    gf.save()
    assert os.path.getmtime("idefix.ini") == time_inifile


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

    with pytest.raises(FileNotFoundError):
        GasDataSet(input_.replace("data.0500", "datawrong.0500"))


def test_api_vtk_by_name_fargo(test_data_dir):
    GasDataSet(test_data_dir / "fargo3d_planet2d" / "gasdens40.dat")


def test_api_fluid_fargo3d(test_data_dir):
    args = (5,)
    kwargs = {
        "fluid": "dust2",
        "directory": test_data_dir / "fargo3d_multifluid",
    }
    ds = GasDataSet(*args, **kwargs)
    assert ds.nfields == 1

    kwargs["fluid"] = "dust999"
    with pytest.raises(
        FileNotFoundError,
        match=r"No file matches the pattern 'dust999\*5\.dat'",
    ):
        GasDataSet(*args, **kwargs)


def test_api_fluid_idefix(test_data_dir):
    with pytest.warns(
        UserWarning,
        match="Unused keyword argument: 'fluid'",
    ):
        GasDataSet(
            500,
            fluid="dust1",
            directory=test_data_dir / "idefix_spherical_planet3d",
        )
