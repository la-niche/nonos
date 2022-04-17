import os
import sys

import pytest

from nonos.api import Parameters
from nonos.config import DEFAULTS
from nonos.main import main

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


@pytest.mark.skipif(sys.platform.startswith("win"), reason="does not run on windows")
def test_config_dir_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"not found"):
        Parameters(inifile="notafile", code="idefix", directory=tmp_path)


def test_config_inifile_but_nocode(tmp_path):
    with pytest.raises(ValueError, match=r"both inifile and code have to be given"):
        Parameters(inifile="notafile", directory=tmp_path)


@pytest.fixture()
def minimal_paramfile(tmp_path):
    ifile = tmp_path / "nonos.toml"
    # check that this setup still makes sense
    assert DEFAULTS["field"] == "RHO"
    with open(ifile, "w") as fh:
        fh.write("field = 'VX1'")
    return ifile


def test_load_config_file(minimal_paramfile, capsys):
    os.chdir(minimal_paramfile.parent)
    ret = main(["-input", "nonos.toml", "-config"])

    assert ret == 0
    out, err = capsys.readouterr()
    assert "Using parameters from" in err
    conf = tomllib.loads(out)
    assert conf["field"] == "VX1"


def test_isolated_mode(minimal_paramfile, capsys):
    os.chdir(minimal_paramfile.parent)
    ret = main(["-config", "-isolated"])

    assert ret == 0
    out, err = capsys.readouterr()
    assert err == ""
    conf = tomllib.loads(out)
    assert conf["field"] == DEFAULTS["field"]
