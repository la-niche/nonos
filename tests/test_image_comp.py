import os

import pytest
from matplotlib.colors import SymLogNorm

from nonos.api import GasDataSet


@pytest.mark.mpl_image_compare()
def test_2D_polar_plane(test_data_dir, temp_figure_and_axis):
    os.chdir(test_data_dir / "idefix_newvtk_planet2d")

    ds = GasDataSet(23)
    fig, ax = temp_figure_and_axis
    ds["VX1"].map("R", "phi").plot(fig, ax, title="vr")
    return fig


@pytest.mark.mpl_image_compare()
def test_symlog(test_data_dir, temp_figure_and_axis):
    os.chdir(test_data_dir / "idefix_newvtk_planet2d")

    ds = GasDataSet(23)
    fig, ax = temp_figure_and_axis
    ds["VX1"].map("R", "phi").plot(
        fig,
        ax,
        title="vr",
        norm=SymLogNorm(vmin=-0.05, vmax=0.05, linthresh=1e-4, base=10),
    )
    return fig


@pytest.mark.mpl_image_compare()
def test_3D_aa_xz(test_data_dir, temp_figure_and_axis):
    os.chdir(test_data_dir / "idefix_spherical_planet3d")

    ds = GasDataSet(500)
    fig, ax = temp_figure_and_axis

    ds["RHO"].azimuthal_average().map("x", "z").plot(fig, ax, log=True, title="rho")
    return fig


@pytest.mark.mpl_image_compare()
def test_3D_vm_phiR(test_data_dir, temp_figure_and_axis):
    os.chdir(test_data_dir / "idefix_spherical_planet3d")

    ds = GasDataSet(500)
    fig, ax = temp_figure_and_axis

    ds["RHO"].vertical_at_midplane().map("phi", "R").plot(
        fig, ax, log=True, title="rho"
    )
    return fig


@pytest.mark.mpl_image_compare()
def test_3D_vm_xy(test_data_dir, temp_figure_and_axis):
    os.chdir(test_data_dir / "idefix_spherical_planet3d")

    ds = GasDataSet(500)
    fig, ax = temp_figure_and_axis

    ds["RHO"].vertical_at_midplane().map("x", "y").plot(fig, ax, log=True, title="rho")
    return fig
