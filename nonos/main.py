#!/usr/bin/env python
"""
Analysis tool for idefix/pluto/fargo3d simulations (in polar coordinates).
"""
# adapted from pbenitez-llambay, gwafflard-fernandez, cmt robert & glesur

import argparse
import functools
import os
import re
import sys
import time
from collections import ChainMap
from multiprocessing import Pool
from typing import Any, Dict, List, Optional

import cblind  # noqa
import inifix
import numpy as np
from inifix.format import iniformat

from nonos.__version__ import __version__
from nonos.api import GasDataSet, Parameters
from nonos.api._angle_parsing import _parse_planet_file
from nonos.config import DEFAULTS
from nonos.logging import (
    configure_logger,
    logger,
    parse_verbose_level,
    print_err,
    print_warn,
)
from nonos.parsing import (
    is_set,
    parse_image_format,
    parse_output_number_range,
    parse_range,
    range_converter,
    userval_or_default,
)
from nonos.styling import set_mpl_style

if sys.version_info >= (3, 9):
    import importlib.resources as importlib_resources
else:
    import importlib_resources


# process function for parallisation purpose with progress bar
# counterParallel = Value('i', 0) # initialization of a counter
def process_field(
    on,
    operations: List[str],
    field,
    plane,
    geometry,
    diff,
    log,
    planet_file: Optional[str],
    extent,
    vmin,
    vmax,
    scaling: float,
    cmap,
    title,
    unit_conversion: int,
    datadir,
    show: bool,
    dpi: int,
    fmt: str,
    theta,
    z,
    phi,
    distance,
    *,
    log_level,
):
    import matplotlib.pyplot as plt

    configure_logger(level=log_level)
    set_mpl_style(scaling=scaling)

    ds = GasDataSet(on, geometry=geometry, directory=datadir)
    dsop = ds[field]
    if diff:
        dsop = dsop.diff(0)
    if "vm" in operations:
        dsop = dsop.vertical_at_midplane()
    elif "vp" in operations:
        dsop = dsop.vertical_projection(z=z)
    elif "lt" in operations:
        dsop = dsop.latitudinal_at_theta(theta=theta)
    elif "lp" in operations:
        dsop = dsop.latitudinal_projection(theta=theta)
    elif "vz" in operations:
        dsop = dsop.vertical_at_z(z=z)

    if "ap" in operations:
        dsop = dsop.azimuthal_at_phi(phi=phi)
    elif "apl" in operations:
        dsop = dsop.azimuthal_at_planet(planet_file=planet_file)
    elif "aa" in operations:
        dsop = dsop.azimuthal_average()

    if "rr" in operations:
        dsop = dsop.radial_at_r(distance=distance)

    logger.debug("operations performed: {}", operations)

    dim = 3 - dsop.shape.count(1)
    logger.debug("plotting a {}D plot.", dim)

    if plane is None:
        dsop_dict = dsop.coords.get_attributes
        default_plane = []
        for key, val in dsop_dict.items():
            if not isinstance(val, str) and val.shape[0] > 2:
                default_plane.append(key)
        # default_plane = ["x","y"]
        plane = default_plane

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=False)
    if dim == 1:
        dsop.map(plane[0], rotate_with=planet_file).plot(
            fig,
            ax,
            log=log,
            vmin=vmin,
            vmax=vmax,
            title="$%s$" % title,
            unit_conversion=unit_conversion,
        )
        akey = dsop.map(plane[0], rotate_with=planet_file).dict_plotable["abscissa"]
        avalue = dsop.map(plane[0], rotate_with=planet_file).dict_plotable[akey]
        extent = parse_range(extent, dim=dim)
        extent = range_converter(extent, abscissa=avalue, ordinate=np.zeros(2))
        ax.set_xlim(extent[0], extent[1])
    elif dim == 2:
        dsop.map(plane[0], plane[1], rotate_with=planet_file).plot(
            fig,
            ax,
            log=log,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            title="$%s$" % title,
            unit_conversion=unit_conversion,
        )
        akey = dsop.map(plane[0], plane[1], rotate_with=planet_file).dict_plotable[
            "abscissa"
        ]
        okey = dsop.map(plane[0], plane[1], rotate_with=planet_file).dict_plotable[
            "ordinate"
        ]
        avalue = dsop.map(plane[0], plane[1], rotate_with=planet_file).dict_plotable[
            akey
        ]
        ovalue = dsop.map(plane[0], plane[1], rotate_with=planet_file).dict_plotable[
            okey
        ]
        extent = parse_range(extent, dim=dim)
        extent = range_converter(
            extent,
            abscissa=avalue,
            ordinate=ovalue,
        )
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    logger.debug("processed the data before plotting.")

    if "x" and "y" in plane:
        ax.set_aspect("equal")

    if show:
        plt.show()
    else:
        logger.debug("saving plot: started")
        filename = f"{''.join(plane)}_{field}_{'_'.join(operations)}{'_diff' if diff else '_'}{'_log' if log else ''}{on:04d}.{fmt}"
        fig.savefig(filename, bbox_inches="tight", dpi=dpi)
        logger.debug("saving plot: finished ({})", filename)

    plt.close(fig)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nonos",
        description=__doc__,
    )

    parser.add_argument(
        "-dir",
        dest="datadir",
        help=f"location of output files and param files (default: '{DEFAULTS['datadir']}').",
    )
    parser.add_argument(
        "-field",
        # choices=["RHO", "VX1", "VX2", "VX3", "BX1", "BX2", "BX3", "PRS"],
        help=f"name of field to plot (default: '{DEFAULTS['field']}').",
    )
    parser.add_argument(
        "-geometry",
        type=str,
        choices=["polar", "cylindrical", "spherical", "cartesian"],
        help=f"if the geometry of idefix outputs is not recognized  (default: '{DEFAULTS['geometry']}').",
    )
    parser.add_argument(
        "-operation",
        type=str,
        nargs="+",
        choices=["vm", "vp", "vz", "lt", "lp", "aa", "ap", "apl", "rr"],
        help=f"operation to apply to the fild (default: '{DEFAULTS['operation']}').",
    )
    parser.add_argument(
        "-plane",
        type=str,
        nargs="+",
        help=f"abscissa and ordinate of the plane of projection (default: '{DEFAULTS['plane']}'), example: r phi",
    )

    # TODO(GWF): add support for -rotate_by -1.2453036989845032 (idefix_newvtk_planet2d)
    parser.add_argument(
        "-corotate",
        type=int,
        default=None,
        help="planet number that defines with which planet the grid corotates.",
    )
    parser.add_argument(
        "-range",
        type=str,
        nargs="+",
        help=f"range of matplotlib window (default: {DEFAULTS['range']}), example: x x -2 2",
    )
    parser.add_argument(
        "-vmin",
        type=float,
        help=f"min value (default: {DEFAULTS['vmin']})",
    )
    parser.add_argument(
        "-vmax",
        type=float,
        help=f"max value (default: {DEFAULTS['vmax']})",
    )
    parser.add_argument(
        "-theta",
        type=float,
        help=f"if latitudinal operation (default: {DEFAULTS['theta']})",
    )
    parser.add_argument(
        "-z",
        type=float,
        help=f"if vertical operation (default: {DEFAULTS['z']})",
    )
    parser.add_argument(
        "-phi",
        type=float,
        help=f"if azimuthal operation (default: {DEFAULTS['phi']})",
    )
    parser.add_argument(
        "-distance",
        type=float,
        help=f"if radial operation (default: {DEFAULTS['distance']})",
    )
    parser.add_argument(
        "-cpu",
        "-ncpu",
        dest="ncpu",
        type=int,
        help=f"number of parallel processes (default: {DEFAULTS['ncpu']}).",
    )

    select_group = parser.add_mutually_exclusive_group()
    select_group.add_argument(
        "-on",
        type=int,
        nargs="+",
        help="output number(s) (on) to plot. "
        "This can be a single value or a range (start, end, [step]) where both ends are inclusive. "
        "(default: last output available).",
    )
    select_group.add_argument(
        "-all",
        action="store_true",
        help="save an image for every available snapshot (this will force show=False).",
    )

    # boolean flags use False as a default value (...by default)
    # forcing them to None instead will allow them to be pruned
    # when we build the ChainMap config
    flag_group = parser.add_argument_group("boolean flags")
    flag_group.add_argument(
        "-diff",
        action="store_true",
        default=None,
        help="plot the relative perturbation of the field f, i.e. (f-f0)/f0.",
    )
    flag_group.add_argument(
        "-log",
        action="store_true",
        default=None,
        help="plot the log10 of the field f, i.e. log(f).",
    )
    flag_group.add_argument(
        "-pbar",
        dest="progressBar",
        action="store_true",
        default=None,
        help="display a progress bar",
    )

    parser.add_argument(
        "-scaling",
        dest="scaling",
        type=float,
        help=f"scale the overall sizes of features in the graph (fonts, linewidth...) (default: {DEFAULTS['scaling']}).",
    )
    parser.add_argument(
        "-cmap",
        help=f"choice of colormap for the 2D maps (default: '{DEFAULTS['cmap']}').",
    )
    parser.add_argument(
        "-title",
        type=str,
        help=f"name of the field in the colorbar for the 2D maps (default: '{DEFAULTS['title']}').",
    )
    parser.add_argument(
        "-uc",
        "-unit_conversion",
        dest="unit_conversion",
        type=float,
        help=f"conversion factor for the considered quantity (default: '{DEFAULTS['unit_conversion']}').",
    )
    parser.add_argument(
        "-fmt",
        "-format",
        dest="format",
        help=f"select output image file format (default: {DEFAULTS['format']})",
    )
    parser.add_argument(
        "-dpi",
        type=int,
        help=f"image file resolution (default: {DEFAULTS['dpi']})",
    )

    cli_only_group = parser.add_argument_group("CLI-only options")
    cli_input_group = cli_only_group.add_mutually_exclusive_group()
    cli_input_group.add_argument(
        "-input", "-i", dest="input", type=str, help="specify a configuration file."
    )
    cli_input_group.add_argument(
        "-isolated", action="store_true", help="ignore any existing 'nonos.ini' file."
    )
    cli_action_group = cli_only_group.add_mutually_exclusive_group()
    cli_action_group.add_argument(
        "-d",
        "-display",
        dest="display",
        action="store_true",
        help="open a graphic window with the plot (only works with a single image)",
    )
    cli_action_group.add_argument(
        "-version",
        "--version",
        action="store_true",
        help="show raw version number and exit",
    )
    cli_action_group.add_argument(
        "-logo",
        action="store_true",
        help="show Nonos logo with version number, and exit.",
    )
    cli_action_group.add_argument(
        "-config", action="store_true", help="show configuration and exit."
    )

    cli_debug_group = cli_only_group.add_mutually_exclusive_group()
    cli_debug_group.add_argument(
        "-v",
        "-verbose",
        "--verbose",
        action="count",
        default=0,
        help="increase output verbosity (-v: info, -vv: debug).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = get_parser()
    clargs = vars(parser.parse_args(argv))

    # special cases: destructively consume CLI-only arguments with dict.pop

    if clargs.pop("logo"):
        logo = importlib_resources.files("nonos").joinpath("logo.txt").read_text()
        print(f"{logo}{__doc__}Version {__version__}")
        return 0

    if clargs.pop("version"):
        print(__version__)
        return 0

    # clargs.pop("verbose")
    level = parse_verbose_level(clargs.pop("verbose"))
    configure_logger(level=level)
    # logger.setLevel(level)

    if clargs.pop("isolated"):
        config_file_args: Dict[str, Any] = {}
    elif (ifile := clargs.pop("input")) is not None:
        if not os.path.isfile(ifile):
            print_err(f"Couldn't find requested input file '{ifile}'.")
            return 1
        print_warn(f"[bold white]Using parameters from '{ifile}'.")
        config_file_args = inifix.load(ifile)
    elif os.path.isfile("nonos.ini"):
        print_warn("[bold white]Using parameters from 'nonos.ini'.")
        config_file_args = inifix.load("nonos.ini")
    else:
        config_file_args = {}

    # check that every parameter in the configuration is also exposed to the CLI
    assert not set(DEFAULTS).difference(set(clargs))

    # squeeze out any unset value form cli config to leave room for file parameters
    clargs = {k: v for k, v in clargs.items() if v is not None}

    # NOTE: init.config is also a ChainMap instance with a default layer
    # this may be seen either as hyperstatism (good thing) or error prone redundancy (bad thing)
    args = ChainMap(clargs, config_file_args, DEFAULTS)
    if clargs.pop("config"):
        conf_repr = {}
        for key in DEFAULTS:
            conf_repr[key] = args[key]
        print(f"# Generated with nonos {__version__}")
        print(iniformat(inifix.dumps(conf_repr)))
        return 0

    try:
        params = Parameters(directory=args["datadir"])
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print_err(exc)
        return 1
    params.loadIniFile()
    params.countSimuFiles()
    data_files = params.data_files

    available = set()
    for fn in data_files:
        if (num := re.search(r"\d+", fn)) is not None:
            available.add(int(num.group()))

    if args.pop("all"):
        requested = available
    else:
        try:
            requested = set(
                parse_output_number_range(args["on"], maxval=max(available))
            )
        except ValueError as exc:
            print_err(exc)
            return 1

    if not (toplot := list(requested.intersection(available))):
        print_err(
            f"No requested output file was found (requested {requested}, found {available})."
        )
        return 1
    args["on"] = toplot

    if (show := clargs.pop("display")) and len(args["on"]) > 1:
        print_warn("display mode can not be used with multiple images, turning it off.")
        show = False

    if not show:
        try:
            args["format"] = parse_image_format(args["format"])
        except ValueError as exc:
            print_err(exc)
            return 1

    # check that every CLI-only argument was consumed at this point
    assert not set(clargs).difference(set(DEFAULTS))

    args["field"] = args["field"].upper()
    extent = args["range"]

    if args["ncpu"] > (ncpu := min(args["ncpu"], os.cpu_count())):
        print_warn(
            f"Requested {args['ncpu']}, but the runner only has access to {ncpu}."
        )

    if args["progressBar"]:
        from rich.progress import track

        def mytrack(iterable, *args, **kwargs):
            return track(iterable, *args, **kwargs)

    else:
        # replace rich.progress.track with a no-op dummy
        def mytrack(iterable, *args, **kwargs):  # noqa: ARG001
            return iterable

    planet_file: Optional[str]
    if not is_set(args["corotate"]):
        planet_file = None
    else:
        planet_file = _parse_planet_file(planet_number=args["corotate"])

    # call of the process_field function, whether it be in parallel or not
    # TODO: reduce this to the bare minimum
    func = functools.partial(
        process_field,
        operations=userval_or_default(args["operation"], default=["vm"]),
        field=args["field"],
        plane=userval_or_default(args["plane"], default=None),
        geometry=userval_or_default(args["geometry"], default="unknown"),
        diff=args["diff"],
        log=args["log"],
        planet_file=planet_file,
        extent=extent,
        vmin=userval_or_default(args["vmin"], default=None),
        vmax=userval_or_default(args["vmax"], default=None),
        scaling=args["scaling"],
        cmap=args["cmap"],
        title=userval_or_default(args["title"], default=args["field"]),
        unit_conversion=args["unit_conversion"],
        datadir=args["datadir"],
        show=show,
        dpi=args["dpi"],
        fmt=args["format"],
        theta=userval_or_default(args["theta"], default=None),
        z=userval_or_default(args["z"], default=None),
        phi=userval_or_default(args["phi"], default=None),
        distance=userval_or_default(args["distance"], default=None),
        log_level=level,
    )

    logger.info("Starting main loop")
    tstart = time.time()
    if "NONOS_DEBUG" in os.environ:
        for on in args["on"]:
            func(on)
    else:
        with Pool(ncpu) as pool:
            list(
                mytrack(
                    pool.imap(func, args["on"]),
                    description="Processing snapshots",
                    total=len(args["on"]),
                )
            )
    if not show:
        logger.info("Operation took {:.2f}s", time.time() - tstart)

    return 0
