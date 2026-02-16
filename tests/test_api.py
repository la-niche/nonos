import os

import numpy as np
import numpy.testing as npt
import pytest

from nonos._geometry import Coordinates, Geometry
from nonos.api import GasDataSet, GasField, file_analysis
from nonos.loaders import Loader


def test_gasfield_immutable_data(test_data_dir):
    arr = np.eye(8)
    field = GasField(
        name="test",
        data=arr,
        coordinates=Coordinates(
            geometry=Geometry.SPHERICAL,
            x1=np.linspace(0, 1, 8),
            x2=np.linspace(0, 1, 8),
            x3=np.linspace(0, 1, 8),
        ),
        native_geometry=Geometry.SPHERICAL,
        output_number=0,
        loader=Loader.resolve(
            directory=test_data_dir / "idefix_planet3d",
        ),
    )
    npt.assert_array_equal(field.data, arr)
    assert field.data is not arr
    with pytest.raises(ValueError, match=r"read-only$"):
        field.data.flat[0] = np.nan


class TestFileAnalysis:
    @pytest.mark.parametrize(
        "directory, expected_shape",
        [("idefix_planet3d", (9, 72)), ("fargo_adsg_planet", (11, 451))],
    )
    def test_simple(self, test_data_dir, directory, expected_shape):
        result = file_analysis(
            "planet0.dat",
            directory=test_data_dir / directory,
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == expected_shape

    def test_norb(self, test_data_dir):
        result = file_analysis(
            "planet0.dat",
            directory=test_data_dir / "idefix_planet3d",
            norb=10,
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (9, 72)

    def test_norb_not_idefix(self, test_data_dir):
        with pytest.raises(NotImplementedError):
            file_analysis(
                "planet0.dat",
                directory=test_data_dir / "fargo_adsg_planet",
                norb=10,
            )

    def test_implicit_directory(self, test_data_dir):
        os.chdir(test_data_dir / "idefix_planet3d")
        result = file_analysis("planet0.dat")
        assert isinstance(result, np.ndarray)
        assert result.shape == (9, 72)


class TestGasDataSetFromNpy:
    expected_keys = ["RHO"]
    args = (7283,)
    kwargs = {"operation": "azimuthal_average"}
    directory = "pluto_spherical"

    def test_from_npy_implicit_directory(self, test_data_dir):
        os.chdir(test_data_dir / self.directory)
        ds = GasDataSet(*self.args, **self.kwargs)
        assert sorted(ds.keys()) == self.expected_keys

    def test_from_npy_explicit_directory(self, test_data_dir):
        ds = GasDataSet(
            *self.args,
            **self.kwargs,
            directory=test_data_dir / self.directory,
        )
        assert sorted(ds.keys()) == self.expected_keys


def test_find_rhill(test_data_dir):
    ds = GasDataSet(23, directory=test_data_dir / "idefix_newvtk_planet2d")
    rp = ds["RHO"].find_rp()
    rhill = ds["RHO"].find_rhill()
    assert rhill < rp


def test_field_map_no_mutation(test_data_dir):
    ds = GasDataSet(500, directory=test_data_dir / "idefix_spherical_planet3d")
    f = ds["RHO"].radial_at_r(1.0).vertical_at_midplane()
    d0 = f.data.copy()
    f.map("phi", rotate_by=1.0)
    d1 = f.data.copy()
    npt.assert_array_equal(d1, d0)
