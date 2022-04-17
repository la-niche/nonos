import os
import sys
import textwrap

from nonos import __version__
from nonos.api import Parameters
from nonos.main import main

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def test_no_inifile(capsys, tmp_path):
    os.chdir(tmp_path)
    ret = main([])
    assert ret != 0
    out, err = capsys.readouterr()
    assert out == ""
    assert err.endswith("idefix.ini, pluto.ini, variables.par not found.\n")


def test_default_conf(capsys, tmp_path):
    os.chdir(tmp_path)
    ret = main(["-config"])
    assert ret == 0
    out, err = capsys.readouterr()
    assert err == ""

    # validate output is reusable
    (tmp_path / "idefix.ini").touch()
    dictout = tomllib.loads(out)
    Parameters(directory=dictout["datadir"])


def test_version(capsys, tmp_path):
    os.chdir(tmp_path)
    ret = main(["-version"])
    assert ret == 0

    out, err = capsys.readouterr()
    assert err == ""
    assert out == str(__version__) + "\n"


def test_logo(capsys, tmp_path):
    os.chdir(tmp_path)
    ret = main(["-logo"])

    assert ret == 0
    out, err = capsys.readouterr()
    expected = textwrap.dedent(
        r"""
                                                                     `!)}$$$$})!`
                 `,!>|))|!,                                        :}&#&$}{{}$&#&};`
              ~)$&&$$}}$$&#$).                                   '}#&}|~'.```.'!($#$+
           `=$&$(^,..``.'~=$&&=                                `|&#}!'`         `'?$#}.
          !$&}:.`         `.!$#$'                             :$#$^'`       ``     .$#$`
         ^&&+.              `'(&$!                          `)&&),`                 !##!
        `$#^      `.   `     `={$&{`                       ^$&$(!'  ``     `,.  ``  !##!
        ,#$ ``                 .>}&${?!:'`   ```..'',:!+|{$&${:`   `'`             `}&}`
        `$$`                       ,;|{}$$$&&&&&$$$$}}()<!,`   '`                 `}&}`
         +&}`   `   |:|.\    |:|            `.```                                  :$$!
          !$$'`!}|  |:|\.\   |:|      __                      __       ___       .{#$
           '$&})$:  |:| \.\  |:|   /./  \.\   |:|.\  |:|   /./  \.\   |:|  \.\  '$$#}
            `}#&;   |:|  \.\ |:|  |:|    |:|  |:|\.\ |:|  |:|    |:|  |:|___     :$)&$`
            `{&!    |:|   \.\|:|  |:|    |:|  |:| \.\|:|  |:|    |:|       |:|    :!{&}`
           :$$,     |:|    \.|:|   \.\__/./   |:|  \.|:|   \.\__/./   \.\__|:|     `:}#}`
          ^&$.                                                                       .$#^
          +&$.                 ``'^)}$$$$$({}}}$$$$$$$$$$}}(|>!.`~:,.                 }#)
         '&#|                 ,|$##$>'`                `'~!)$##$$$)?^,`           ` :&&:
         ,&#}`  ``       .` `:{$&}:                          ~}&$)^^+=^`  `` ..  .|&#)
          |&#$:```   `` '::!}$&},                              !$&$|++^^:,:~',!!($#&^
           ,}&#${^~,,,:!|}$&&(.                                  ^$#$}{)|?|)(}$&#$?`
             :{&##$$$$$&##$).                                      ~($&#&&##&$}=,
               `:|}$$$$}):`

        Analysis tool for idefix/pluto/fargo3d simulations (in polar coordinates).
        """.lstrip(
            "\n"
        )
    )
    expected += f"Version {__version__}\n"
    assert out == expected
    assert err == ""
