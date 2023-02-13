import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from lick.lick import lick_box
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage.filters import uniform_filter1d

from nonos.api.analysis import GasField, Plotable, from_data, temporal
from nonos.api.from_simulation import Parameters
from nonos.logging import logger


def file_analysis(filename, *, inifile="", code="", directory="", norb=None):
    fullpath = os.path.join(directory, filename)
    with open(fullpath) as f1:
        data = f1.readlines()
    y = [[v for v in re.split(r"[\t ]+", r)] for r in data]
    columns = np.array(y, dtype="float64").T
    if norb is not None:
        init = Parameters(inifile=inifile, code=code, directory=directory)
        init.loadIniFile()
        if init.code == "idefix" and "analysis" in init.inifile["Output"].keys():
            analysis = init.inifile["Output"]["analysis"]
            rpini = init.inifile["Planet"]["initialDistance"]
            Ntmean = round(norb * 2 * np.pi * pow(rpini, 1.5) / analysis)
            for i in range(1, len(columns) - 1):
                columns[i] = uniform_filter1d(columns[i], Ntmean)
        else:
            raise NotImplementedError(
                f"moving average on {norb} orbits is not implemented for the code {init.code}"
            )
    return columns


def planet_analysis(planet_number, *, inifile="", code="", directory="", norb=None):
    init = Parameters(inifile=inifile, code=code, directory=directory)
    init.loadIniFile()
    init.loadPlanetFile(planet_number=planet_number)
    return init


def save_temporal(
    field: str,
    onbeg: int,
    onend: int,
    operation: str,
    *,
    directory: str = "",
    planet_corotation=0,
):
    temporal_evolution = temporal(
        field,
        operation,
        onbeg,
        onend=onend,
        directory=directory,
        planet_corotation=planet_corotation,
    ).save(directory=directory)
    return temporal_evolution


def load_fields(
    field_operation: Dict[str, Any],
    fields: List[str],
    operations: List[Tuple[str, str]],
    onbeg: int,
    onend: int,
    *,
    directory: str = "",
    temporal_bool: bool = True,
    snapshot_bool: bool = True,
    planet_corotation=0,
):
    for field in fields:
        for operation, operation_shortcut in operations:
            if snapshot_bool:
                field_operation[f"{field}{operation_shortcut}"] = temporal(
                    field,
                    operation,
                    onend,
                    onend=None,
                    directory=directory,
                    planet_corotation=planet_corotation,
                )
            if temporal_bool:
                try:
                    field_operation[f"{field}T{operation_shortcut}"] = temporal(
                        f"{field}T_{onbeg}_{onend}",
                        operation,
                        onend,
                        onend=None,
                        directory=directory,
                    )
                except FileNotFoundError:
                    save_temporal(
                        field,
                        onbeg,
                        onend,
                        operation,
                        directory=directory,
                        planet_corotation=planet_corotation,
                    )
                    field_operation[f"{field}T{operation_shortcut}"] = temporal(
                        f"{field}T_{onbeg}_{onend}",
                        operation,
                        onend,
                        onend=None,
                        directory=directory,
                    )
    return field_operation


class NonosLick:
    def __init__(
        self,
        dirline1: str,
        dirline2: str,
        field: str,
        line1: str,
        line2: str,
        onbeg: int,
        onend: int,
        operation: Optional[str] = None,
        directory: str = "",
        temporal_bool: bool = True,
        planet_corotation=0,
    ):
        if operation is None:
            raise ValueError(
                f"You need to choose an operation to perform on {field}, {line1} and {line2}."
            )
        self.temporal_bool = temporal_bool
        self.snapshot_bool = not (self.temporal_bool)

        # print("find names --> done")

        field_operation: Dict[str, Any] = {}
        fields = [field, line1, line2]
        # find an abreviation of operation by considering the first character after each "_" (ex: "vertical_at_midplane" -> "VAM")
        operations = [
            (
                operation,
                "".join(
                    [
                        operation[0],
                        "".join(
                            [
                                operation[pos + 1]
                                for pos, char in enumerate(operation)
                                if char == "_"
                            ]
                        ),
                    ]
                ).upper(),
            )
        ]
        # print("--> done")

        self.fields = fields
        self.operation_shortcut = operations[0][1]

        # print("load fields")

        self.field_operation = load_fields(
            field_operation,
            fields,
            operations,
            onbeg,
            onend,
            directory=directory,
            snapshot_bool=self.snapshot_bool,
            temporal_bool=self.temporal_bool,
            planet_corotation=planet_corotation,
        )
        # print("--> done")

        # print("rename fields + gather in dictionary")
        if self.temporal_bool:
            self.background = self.field_operation[
                f"{self.fields[0]}T{self.operation_shortcut}"
            ]
            # self.lines = (self.field_operation[f"{self.fields[1]}T{self.operation_shortcut}"], self.field_operation[f"{self.fields[2]}T{self.operation_shortcut}"])
            self.lines = [
                self.field_operation[f"{self.fields[i]}T{self.operation_shortcut}"]
                for i in (1, 2)
            ]
        else:
            self.background = self.field_operation[
                f"{self.fields[0]}{self.operation_shortcut}"
            ]
            # self.lines = (self.field_operation[f"{self.fields[1]}{self.operation_shortcut}"], self.field_operation[f"{self.fields[2]}{self.operation_shortcut}"])
            self.lines = [
                self.field_operation[f"{self.fields[i]}{self.operation_shortcut}"]
                for i in (1, 2)
            ]
        # print("--> done")

        self.abscissa_str = dirline1
        self.ordinate_str = dirline2
        self.abscissa = self.background.coords.get_coords["".join([dirline1, "med"])]
        self.ordinate = self.background.coords.get_coords["".join([dirline2, "med"])]

    def load(
        self,
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
        ymin: Optional[float] = None,
        ymax: Optional[float] = None,
    ):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        # print("preprocess the variables to fit the lick_box constraints")
        position_of_3d_dimension = self.background.shape.index(1)
        ordered = self.background.coords._meshgrid_conversion(
            self.abscissa_str, self.ordinate_str
        )["ordered"]
        if position_of_3d_dimension == 1:
            ordered = not ordered
        if ordered:
            data_background = np.moveaxis(
                self.background.data, position_of_3d_dimension, 0
            )[0]
            # data_lines = (np.moveaxis(self.lines[0].data, position_of_3d_dimension, 0)[0], np.moveaxis(self.lines[1].data, position_of_3d_dimension, 0)[0])
            data_lines = [
                np.moveaxis(self.lines[i].data, position_of_3d_dimension, 0)[0]
                for i in (0, 1)
            ]
        else:
            raise NotImplementedError(
                f"Not yet fully implemented, try {self.ordinate_str,self.abscissa_str} instead"
            )
            # data_background = np.moveaxis(
            #     self.background.data, position_of_3d_dimension, 0
            # )[0].T
            # # data_lines = (np.moveaxis(self.lines[0].data, position_of_3d_dimension, 0)[0].T, np.moveaxis(self.lines[1].data, position_of_3d_dimension, 0)[0].T)
            # data_lines = [
            #     np.moveaxis(self.lines[i].data, position_of_3d_dimension, 0)[0].T
            #     for i in (0, 1)
            # ]

        if "VX2" in self.fields[1:]:
            logger.info(
                "Warning: for now, we suppose that the planet is in a fixed circular orbit at R = 1"
            )
            index_vphi = self.fields[1:].index("VX2")
            if ordered:
                data_lines[index_vphi] -= self.lines[index_vphi].coords.Rmed[:, None]
            else:
                data_lines[index_vphi] -= self.lines[index_vphi].coords.Rmed[None, :]

        # print("--> done")

        # print("lick_box")
        self.X, self.Y, self.LINE1, self.LINE2, self.F, self.lick = lick_box(
            self.abscissa,
            self.ordinate,
            data_lines[0],
            data_lines[1],
            data_background,
            xmin=self.xmin,
            xmax=self.xmax,
            ymin=self.ymin,
            ymax=self.ymax,
            size_interpolated=1000,
            niter_lic=6,
            kernel_length=101,
        )
        # print("--> done")

        # print("create two dictionaries for the interpolated background + the lick itself")
        dict_lick = {}
        dict_background = {}
        # print("--> done")

        # print("fill both dictionaries with the correct items")
        dict_background["field"] = self.fields[0]
        dict_background["abscissa"] = self.abscissa_str
        dict_background["ordinate"] = self.ordinate_str
        dict_background[dict_background["field"]] = self.F
        dict_background[dict_background["abscissa"]] = self.X
        dict_background[dict_background["ordinate"]] = self.Y

        dict_lick["field"] = "lick"
        dict_lick["abscissa"] = self.abscissa_str
        dict_lick["ordinate"] = self.ordinate_str
        dict_lick[dict_lick["field"]] = self.lick
        dict_lick[dict_lick["abscissa"]] = self.X
        dict_lick[dict_lick["ordinate"]] = self.Y
        # print("--> done")

        self.dict_background = dict_background
        self.dict_lick = dict_lick

    def plot(
        self,
        fig,
        ax,
        *,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        alpha: float = 0.03,
        log: bool = False,
        cmap=None,
        title: Optional[str] = None,
        density_streamlines: Optional[float] = None,
        **kwargs,
    ):
        im = Plotable(self.dict_background).plot(
            fig,
            ax,
            vmin=vmin,
            vmax=vmax,
            log=log,
            cmap=cmap,
            filename=None,
            dpi=500,
            title=title,
            shading="nearest",
            rasterized=True,
            edgecolor="face",
        )
        Plotable(self.dict_lick).plot(
            fig,
            ax,
            # vmin=vmin,
            # vmax=vmax,
            log=False,
            cmap="binary_r",
            filename=None,
            dpi=500,
            title=None,
            alpha=alpha,
            shading="nearest",
            rasterized=True,
            edgecolor="face",
        )
        if density_streamlines is not None:
            ax.streamplot(
                self.X,
                self.Y,
                self.LINE1,
                self.LINE2,
                density=density_streamlines,
                arrowstyle="->",
                linewidth=0.5,
                color="k",
                # color=np.log10(self.F*np.sqrt(self.LINE1**2+self.LINE2**2)),#/np.max(np.log10(self.F*np.sqrt(self.LINE1**2+self.LINE2**2))),
                # cmap=cb.cbmap("binary_r"),
            )
        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(self.ymin, self.ymax)
        ax.set_xlabel(self.abscissa_str)
        ax.set_ylabel(self.ordinate_str)
        return im


class NonosSpacetime:
    def __init__(
        self,
        field: str,
        operation: str,
        onarray,
        *,
        distance: Optional[float] = None,
        directory: str = "",
    ):
        self.directory = directory
        self.field = field
        self.operation = operation
        self.onarray = np.array(sorted(onarray))
        self.distance = distance

    def load(self, *, log: bool = False, planet_corotation=0):
        if self.distance is None:
            fieldvpaa0 = temporal(
                self.field,
                self.operation,
                self.onarray[0],
                onend=None,
                directory=self.directory,
                planet_corotation=planet_corotation,
            ).azimuthal_average()
            self.direction = fieldvpaa0.coords.Rmed
            self.spacetime = np.zeros((len(self.onarray), fieldvpaa0.shape[0]))
            count = 0
            for on in self.onarray:
                self.fieldvpaa = temporal(
                    self.field,
                    self.operation,
                    on,
                    onend=None,
                    directory=self.directory,
                    planet_corotation=planet_corotation,
                ).azimuthal_average()
                if log:
                    self.spacetime[count, :] = np.log10(self.fieldvpaa.data[:, 0, 0])
                else:
                    self.spacetime[count, :] = self.fieldvpaa.data[:, 0, 0]
                count += 1
        else:
            if self.distance >= 0:
                fieldvpaa0 = temporal(
                    self.field,
                    self.operation,
                    self.onarray[0],
                    onend=None,
                    directory=self.directory,
                    planet_corotation=planet_corotation,
                ).radial_at_r(distance=self.distance)
                self.direction = fieldvpaa0.coords.zmed
                self.spacetime = np.zeros((len(self.onarray), fieldvpaa0.shape[2]))
                count = 0
                for on in self.onarray:
                    self.fieldvpaa = temporal(
                        self.field,
                        self.operation,
                        on,
                        onend=None,
                        directory=self.directory,
                        planet_corotation=planet_corotation,
                    ).radial_at_r(distance=self.distance)
                    if log:
                        self.spacetime[count, :] = np.log10(
                            self.fieldvpaa.data[0, 0, :]
                        )
                    else:
                        self.spacetime[count, :] = self.fieldvpaa.data[0, 0, :]
                    count += 1
            else:
                # raise NotImplementedError(
                #     "Slice at R=-distance & phi-t spacetime diagram not implemented yet."
                # )
                fieldvpaa0 = temporal(
                    self.field,
                    self.operation,
                    self.onarray[0],
                    onend=None,
                    directory=self.directory,
                    planet_corotation=planet_corotation,
                ).radial_at_r(distance=-self.distance)
                self.direction = fieldvpaa0.coords.phimed
                self.spacetime = np.zeros((len(self.onarray), fieldvpaa0.shape[1]))
                count = 0
                for on in self.onarray:
                    self.fieldvpaa = temporal(
                        self.field,
                        self.operation,
                        on,
                        onend=None,
                        directory=self.directory,
                        planet_corotation=planet_corotation,
                    ).radial_at_r(distance=-self.distance)
                    if log:
                        self.spacetime[count, :] = np.log10(
                            self.fieldvpaa.data[0, :, 0]
                        )
                    else:
                        self.spacetime[count, :] = self.fieldvpaa.data[0, :, 0]
                    count += 1

    def save(self):
        from_data(
            field="SPACETIME",
            data=np.array(self.spacetime),
            coords=self.fieldvpaa.coords,
            on=0,
            operation=self.fieldvpaa.operation,
            directory=self.directory,
            rotate_grid=self.fieldvpaa.rotate_grid,
        ).save(directory=self.directory)

    def plot(
        self,
        fig,
        ax,
        *,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap=None,
        title: Optional[str] = None,
    ):
        if vmin is None:
            vmin = self.spacetime.min()
        if vmax is None:
            vmax = self.spacetime.max()
        if self.distance is None:
            im = ax.pcolormesh(
                self.onarray / 4.0,
                self.direction,
                self.spacetime.T,
                shading="nearest",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
            ax.set_xlabel("t [orbs]")
            ax.set_ylabel("R [c.u]")
        else:
            if self.distance >= 0:
                im = ax.pcolormesh(
                    self.onarray / 4.0,
                    self.direction,
                    self.spacetime.T,
                    shading="nearest",
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                )
                ax.set_xlabel("t [orbs]")
                ax.set_ylabel("z [c.u]")
            else:
                x, y = np.meshgrid(self.onarray / 4.0, self.direction, indexing="ij")
                X, Y = (x * np.cos(y), x * np.sin(y))
                im = ax.pcolormesh(
                    X,
                    Y,
                    self.spacetime,
                    shading="nearest",
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                )
                # im = ax.pcolormesh(self.onarray/4., self.direction, self.spacetime.T, shading="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
                # ax.set_ylabel("phi [rad]")
                ax.set_xlabel("t [orbs]")
                ax.set_ylabel("t [orbs]")
                ax.set_aspect("equal")
        if title is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation="vertical")  # , format='%.0e')
            cbar.set_label(title)


def compute(
    field: str,
    data: np.ndarray,
    ref: GasField,
    *,
    inifile: str = "",
    code: str = "",
    directory: str = "",
    rotate_grid: bool = False,
):
    ret_data = data
    ret_coords = ref.coords
    geometry = ret_coords.geometry
    return GasField(
        field,
        ret_data,
        ret_coords,
        geometry,
        ref.on,
        operation=ref.operation,
        inifile=inifile,
        code=code,
        directory=directory,
        rotate_grid=rotate_grid,
    )
