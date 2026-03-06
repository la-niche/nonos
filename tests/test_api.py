import operator
import os
from math import prod

import numpy as np
import numpy.testing as npt
import pytest
from pytest import RaisesExc, RaisesGroup

from nonos._geometry import (
    SELECT_ALL,
    AutoIndex,
    Axis,
    Coordinates,
    Geometry,
    IndexRange,
)
from nonos.api import GasDataSet, file_analysis
from nonos.api.analysis import Field


@pytest.fixture
def stub_field():
    return Field(
        name="test",
        data=np.arange(1, 31, dtype="float64").reshape(2, 3, 5),
        coordinates=Coordinates(
            geometry=Geometry.CARTESIAN,
            x1=np.linspace(0, 1, 3),
            x2=np.linspace(0, 1, 4),
            x3=np.linspace(0, 1, 6),
        ),
    )


def test_field_immutable_data(stub_field):
    arr = np.arange(30, dtype="float64").reshape(2, 3, 5)
    field = stub_field.replace(data=arr)
    npt.assert_array_equal(field.data, arr)

    assert field.data is not arr
    with pytest.raises(ValueError, match=r"read-only$"):
        field.data.flat[0] = np.nan


@pytest.mark.parametrize(
    "newtype, ctx",
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
def test_field_mismatch_dtypes(stub_field, newtype, ctx):
    arr = stub_field.as_3dview().astype(newtype)
    with ctx:
        stub_field.replace(data=arr)


@pytest.mark.parametrize(
    "shape, effective_ndim",
    [
        ((2, 3, 5), 3),
        ((1, 3, 5), 2),
        ((2, 1, 5), 2),
        ((2, 3, 1), 2),
        ((1, 1, 5), 1),
        ((1, 3, 1), 1),
        ((2, 1, 1), 1),
        ((1, 1, 1), 0),
    ],
)
def test_field_ndviews(stub_field, shape, effective_ndim, subtests):
    size = prod(shape)
    arr = np.arange(size, dtype="float64").reshape(shape)

    field = stub_field.replace(
        data=arr,
        coordinates=Coordinates(
            geometry=Geometry.CARTESIAN,
            x1=np.linspace(0, 1, shape[0] + 1),
            x2=np.linspace(0, 1, shape[1] + 1),
            x3=np.linspace(0, 1, shape[2] + 1),
        ),
    )
    allowed_dims = sorted({effective_ndim, 3})
    for ndim in allowed_dims:
        method_name = f"as_{ndim}dview"
        assert hasattr(field, method_name)
        with subtests.test(ndim=ndim):
            v1 = field.as_ndview(ndim=ndim)
            v2 = getattr(field, method_name)()
            assert v1.ndim == ndim
            npt.assert_array_equal(v1.squeeze(), arr.squeeze())
            npt.assert_array_equal(v2, v1)
            assert not v1.flags.writeable
            assert not v2.flags.writeable

    forbidden_dims = sorted({0, 1, 2, 3}.difference(allowed_dims))
    for ndim in forbidden_dims:
        method_name = f"as_{ndim}dview"
        assert hasattr(field, method_name)
        if effective_ndim < ndim < 3:
            msg = (
                rf"^Cannot produce a {ndim}d view from effective ndim {effective_ndim}$"
            )
        else:
            msg = rf"^Effective ndim {effective_ndim} is greater than target {ndim}$"
        subctx = pytest.raises(TypeError, match=msg)
        with subtests.test(ndim=ndim):
            with subctx:
                field.as_ndview(ndim=ndim)
            with subctx:
                getattr(field, method_name)()

    assert not hasattr(field, "as_4dview")
    with (
        subtests.test(ndim=4),
        pytest.raises(
            ValueError,
            match=r"^Expected ndim to be either 0, 1, 2 or 3\. Got ndim=4$",
        ),
    ):
        field.as_ndview(ndim=4)


@pytest.mark.parametrize(
    "op", [operator.add, operator.sub, operator.mul, operator.truediv]
)
@pytest.mark.parametrize(
    "operand",
    ["COPY", 2, 3.0, np.float32(4), np.float64(5), 1.0 + 0j, np.complex128(2.0 + 0.0j)],
)
def test_field_arithemtic(stub_field, op, operand):
    if operand == "COPY":
        # a shallow copy
        operand = stub_field.replace()

    if isinstance(operand, complex):
        with pytest.raises(TypeError):
            op(stub_field, operand)
        with pytest.raises(TypeError):
            op(operand, stub_field)
        return

    res1 = op(stub_field, operand)
    res2 = op(operand, stub_field)
    assert res1.dtype == stub_field.dtype
    assert res2.dtype == stub_field.dtype

    match operand:
        case Field():
            true_operand = operand.data
        case _:
            true_operand = operand
    npt.assert_array_equal(res1.as_3dview(), op(stub_field.data, true_operand))
    npt.assert_array_equal(res2.as_3dview(), op(true_operand, stub_field.data))


@pytest.mark.parametrize("axis", [Axis.CARTESIAN_X, Axis.CARTESIAN_Y, Axis.CARTESIAN_Z])
@pytest.mark.parametrize(
    "idx", [0, -1, -2, 1, AutoIndex.LEFTMOST, AutoIndex.RIGHTMOST, AutoIndex.MIDPOINT]
)
@pytest.mark.parametrize("keep_name", [True, False])
def test_field_slice(stub_field, axis, idx, keep_name):
    f = stub_field
    assert f.effective_ndim == 3

    s = f.slice_at_index(axis, idx, keep_name=keep_name)
    assert s.effective_ndim == 2
    if keep_name:
        assert s.name == f.name
    else:
        assert s.name == "<slice-result>"

    assert s.slice_at_index(axis, AutoIndex.MIDPOINT, keep_name=True) == s


@pytest.mark.parametrize(
    "selection, expected_ndim",
    [
        (
            {
                Axis.CARTESIAN_X: SELECT_ALL,
                Axis.CARTESIAN_Y: SELECT_ALL,
                Axis.CARTESIAN_Z: SELECT_ALL,
            },
            3,
        ),
        # TODO: re-implement slicing as an application of this
        ({Axis.CARTESIAN_Y: IndexRange(0, 1)}, 2),
    ],
)
@pytest.mark.parametrize("keep_name", [True, False])
def test_field_select_subdomain(stub_field, selection, expected_ndim, keep_name):
    f = stub_field
    assert f.effective_ndim == 3

    s = f.select_subdomain(selection, keep_name=keep_name)
    assert s.effective_ndim == expected_ndim
    if keep_name:
        assert s.name == f.name
    else:
        assert s.name == "<select-result>"


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

    def test_norb_not_uidefix(self, test_data_dir):
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
