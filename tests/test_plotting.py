import os
import re
from glob import glob
from pathlib import Path

import pytest

from nonos.main import main

@pytest.fixture()
def test_data_dir():
    return Path(__file__).parent / "data"

@pytest.fixture(params=["idefix_rwi", "idefix_planet3d"])
def simulation_dir(test_data_dir, request):
    return test_data_dir / request.param

ARGS_TO_CHECK = {
    "vanilla_conf": [],
    "diff": ["-diff"],
    "log": ["-log"],
    "movie": ["-all", "-pol"],
    "movie_with_diff": ["-all", "-diff"],
    "movie_with_multiproc": ["-all", "-ncpu", "2"],
}

@pytest.mark.parametrize("argv", ARGS_TO_CHECK.values(), ids=ARGS_TO_CHECK.keys())
def test_plot_simple(argv, simulation_dir, capsys, tmp_path):
    os.chdir(tmp_path)
    ret = main(argv + ["-dir", str(simulation_dir)])

    out, err = capsys.readouterr()
    assert err == ""
    assert re.match(r"Operation took \d+.\d\ds\n", out)
    assert ret == 0
    assert len(glob("*.png")) > 0

@pytest.mark.parametrize("format", ["pdf", "png", "jpg"])
def test_plot_simple(format, simulation_dir, capsys, tmp_path):
    os.chdir(tmp_path)
    ret = main(["-dir", str(simulation_dir), "-fmt", format])

    out, err = capsys.readouterr()
    assert err == ""
    assert re.match(r"Operation took \d+.\d\ds\n", out)
    assert ret == 0
    assert len(glob(f"*.{format}")) == 1

def test_plot_simple_corotation(simulation_dir, capsys):
    # just check that the call returns no err
    ret = main(["-cor", "-dir", str(simulation_dir)])

    out, err = capsys.readouterr()
    assert re.match(r"Operation took \d+.\d\ds\n", out)
    # ignore differences in text wrapping because they are an implementation detail
    # due to the fact we use rich to display warnings
    assert err.strip().replace("\n", " ").endswith("We don't rotate the grid if there is no planet for now. omegagrid = 0.")
    assert ret == 0

def test_plot_memory(simulation_dir, capsys):
    ret = main(["-memory", "-dir", str(simulation_dir)])

    out, err = capsys.readouterr()
    assert re.match(r"Operation took \d+.\d\ds\nCurrent memory usage is \d+.\d\dMB; Peak was \d+.\d\dMB", out)
    assert err == ""
    assert ret == 0

def test_plot_planet_corotation(test_data_dir):
    from nonos import InitParamNonos, FieldNonos
    from nonos.main import find_nearest

    os.chdir(test_data_dir / "idefix_planet3d") 

    init = InitParamNonos()
    fieldon = FieldNonos(init, field='RHO', on=43)
    azimfield=fieldon.data[find_nearest(fieldon.x,1.0),:,fieldon.imidplane]
    assert find_nearest(azimfield, azimfield.max()) != fieldon.ny//2

    initPlanet = InitParamNonos(isPlanet=True, corotate=True)
    fieldonPlanet = FieldNonos(initPlanet, field='RHO', on=43)
    azimfieldPlanet = fieldonPlanet.data[find_nearest(fieldonPlanet.x,1.0),:,fieldonPlanet.imidplane]
    assert find_nearest(azimfieldPlanet, azimfieldPlanet.max()) == fieldonPlanet.ny//2
