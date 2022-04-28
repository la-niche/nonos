import os

import pytest
from matplotlib.colors import SymLogNorm

from nonos.api import GasDataSet


@pytest.mark.mpl_image_compare()
def test_2D_polar_plane(test_data_dir, temp_figure_and_axis):
    curdir = os.path.abspath(os.curdir)
    os.chdir(test_data_dir / "idefix_newvtk_planet2d")

    ds = GasDataSet(23)
    fig, ax = temp_figure_and_axis
    ds["VX1"].map("R", "phi").plot(fig, ax, title="vr")
    os.chdir(curdir)
    return fig


@pytest.mark.mpl_image_compare()
def test_symlog(test_data_dir, temp_figure_and_axis):
    curdir = os.path.abspath(os.curdir)
    os.chdir(test_data_dir / "idefix_newvtk_planet2d")

    ds = GasDataSet(23)
    fig, ax = temp_figure_and_axis
    ds["VX1"].map("R", "phi").plot(
        fig,
        ax,
        title="vr",
        norm=SymLogNorm(vmin=-0.05, vmax=0.05, linthresh=1e-4, base=10),
    )
    os.chdir(curdir)
    return fig
