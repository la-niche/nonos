import dataclasses
import json
import warnings
from collections.abc import ItemsView, KeysView, ValuesView
from functools import cached_property
from pathlib import Path
from shutil import copyfile
from typing import TYPE_CHECKING, Any, Optional, Union, overload

import numpy as np
from matplotlib.scale import SymmetricalLogTransform
from matplotlib.ticker import SymmetricalLogLocator

from nonos._readers.binary import NPYReader
from nonos._types import PathT, PlanetData
from nonos.api._angle_parsing import (
    _fequal,
    _parse_planet_file,
    _parse_rotation_angle,
)
from nonos.api.tools import find_around, find_nearest
from nonos.loaders import Recipe, loader_from, recipe_from
from nonos.logging import logger

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class Plotable:
    def __init__(self, dict_plotable: dict) -> None:
        self.dict_plotable = dict_plotable
        self.data = self.dict_plotable[self.dict_plotable["field"]]
        self.dimension = len(self.data.shape)
        if self.dimension > 2:
            raise TypeError(
                "Plotable doesn't support data with dimensionality>2, "
                f"got {self.dimension}"
            )

    def plot(
        self,
        fig: "Figure",
        ax: "Axes",
        *,
        log=False,
        cmap="inferno",
        filename=None,
        fmt="png",
        dpi=500,
        title=None,
        unit_conversion=None,
        nbin=None,  # deprecated
        **kwargs,
    ):
        if nbin is not None:
            warnings.warn(
                "The nbin parameter has no effect and is deprecated",
                stacklevel=2,
            )
        data = self.data
        if unit_conversion is not None:
            data = data * unit_conversion
        if log:
            data = np.log10(data)

        if self.dimension == 2:
            self.akey = self.dict_plotable["abscissa"]
            self.okey = self.dict_plotable["ordinate"]
            self.avalue = self.dict_plotable[self.akey]
            self.ovalue = self.dict_plotable[self.okey]
            kw = {}
            if (norm := kwargs.get("norm")) is not None:
                if "vmin" in kwargs:
                    norm.vmin = kwargs.pop("vmin")
                if "vmax" in kwargs:
                    norm.vmax = kwargs.pop("vmax")
            else:
                vmin = kwargs.pop("vmin") if "vmin" in kwargs else np.nanmin(data)
                vmax = kwargs.pop("vmax") if "vmax" in kwargs else np.nanmax(data)
                kw.update({"vmin": vmin, "vmax": vmax})

            im = ax.pcolormesh(
                self.avalue,
                self.ovalue,
                data,
                cmap=cmap,
                **kwargs,
                **kw,
            )
            ax.set(
                xlim=(self.avalue.min(), self.avalue.max()),
                ylim=(self.ovalue.min(), self.ovalue.max()),
            )

            ax.set_xlabel(self.akey)
            ax.set_ylabel(self.okey)
            if title is not None:
                from mpl_toolkits.axes_grid1 import make_axes_locatable

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(
                    im, cax=cax, orientation="vertical"
                )  # , format='%.0e')
                cbar.set_label(title)

                cb_axis = cbar.ax.yaxis
                if cb_axis.get_scale() == "symlog":
                    # no minor tick is drawn in symlog norms by default
                    # as of matplotlib 3.7.1, see
                    # https://github.com/matplotlib/matplotlib/issues/25994
                    trf = cb_axis.get_transform()
                    if not isinstance(trf, SymmetricalLogTransform):
                        raise RuntimeError
                    cb_axis.set_major_locator(SymmetricalLogLocator(trf))
                    if float(trf.base).is_integer():
                        locator = SymmetricalLogLocator(
                            trf, subs=list(range(1, int(trf.base)))
                        )
                        cb_axis.set_minor_locator(locator)
            else:
                return im
        elif self.dimension == 1:
            vmin = kwargs.pop("vmin") if "vmin" in kwargs else np.nanmin(data)
            vmax = kwargs.pop("vmax") if "vmax" in kwargs else np.nanmax(data)
            self.akey = self.dict_plotable["abscissa"]
            self.avalue = self.dict_plotable[self.akey]
            if "norm" in kwargs:
                logger.info("norm has no meaning in 1D.")
                kwargs.pop("norm")
            ax.plot(self.avalue, data, **kwargs)
            ax.set_ylim(ymin=vmin)
            ax.set_ylim(ymax=vmax)
            ax.set_xlabel(self.akey)
            if title is not None:
                ax.set_ylabel(title)
        else:
            raise TypeError(
                "Plotable doesn't support data with dimensionality>2, "
                f"got {self.dimension}"
            )
        if filename is not None:
            fig.savefig(f"{filename}.{fmt}", bbox_inches="tight", dpi=dpi)


class Coordinates:
    """Coordinates class from x1, x2, x3"""

    def __init__(
        self, geometry: str, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray
    ) -> None:
        if x1.shape[0] == 1:
            x1 = np.array([x1[0], x1[0]])
        if x2.shape[0] == 1:
            x2 = np.array([x2[0], x2[0]])
        if x3.shape[0] == 1:
            x3 = np.array([x3[0], x3[0]])
        self.geometry = geometry
        if self.geometry == "cartesian":
            self.x = x1
            self.y = x2
            self.z = x3
            self.cube = ("x", "y", "z")
            self.xmed = 0.5 * (self.x[1:] + self.x[:-1])
            self.ymed = 0.5 * (self.y[1:] + self.y[:-1])
            self.zmed = 0.5 * (self.z[1:] + self.z[:-1])
        if self.geometry == "spherical":
            self.r = x1
            self.theta = x2
            self.phi = x3
            self.cube = ("r", "theta", "phi")
            self.rmed = 0.5 * (self.r[1:] + self.r[:-1])
            self.thetamed = 0.5 * (self.theta[1:] + self.theta[:-1])
            self.phimed = 0.5 * (self.phi[1:] + self.phi[:-1])
        if self.geometry == "polar":
            self.R = x1
            self.phi = x2
            self.z = x3
            self.cube = ("R", "phi", "z")
            self.Rmed = 0.5 * (self.R[1:] + self.R[:-1])
            self.phimed = 0.5 * (self.phi[1:] + self.phi[:-1])
            self.zmed = 0.5 * (self.z[1:] + self.z[:-1])

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        Returns
        =======
        shape : tuple
        """
        if self.geometry == "cartesian":
            return len(self.x), len(self.y), len(self.z)
        elif self.geometry == "spherical":
            return len(self.r), len(self.theta), len(self.phi)
        elif self.geometry == "polar":
            return len(self.R), len(self.phi), len(self.z)
        else:
            raise RuntimeError(f"Unknown geometry {self.geometry!r}")

    @property
    def get_attributes(self) -> dict[str, Any]:
        if self.geometry == "cartesian":
            return {"geometry": self.geometry, "x": self.x, "y": self.y, "z": self.z}
        elif self.geometry == "spherical":
            return {
                "geometry": self.geometry,
                "r": self.r,
                "theta": self.theta,
                "phi": self.phi,
            }
        elif self.geometry == "polar":
            return {
                "geometry": self.geometry,
                "R": self.R,
                "phi": self.phi,
                "z": self.z,
            }
        else:
            raise RuntimeError(f"Unknown geometry {self.geometry!r}")

    @property
    def get_coords(self) -> dict[str, Any]:
        if self.geometry == "cartesian":
            return {
                "x": self.x,
                "y": self.y,
                "z": self.z,
                "xmed": self.xmed,
                "ymed": self.ymed,
                "zmed": self.zmed,
            }
        elif self.geometry == "spherical":
            return {
                "r": self.r,
                "theta": self.theta,
                "phi": self.phi,
                "rmed": self.rmed,
                "thetamed": self.thetamed,
                "phimed": self.phimed,
            }
        elif self.geometry == "polar":
            return {
                "R": self.R,
                "phi": self.phi,
                "z": self.z,
                "Rmed": self.Rmed,
                "phimed": self.phimed,
                "zmed": self.zmed,
            }
        else:
            raise RuntimeError(f"Unknown geometry {self.geometry!r}")

    def _meshgrid_reduction(self, *reducted) -> dict:
        for i in reducted:
            if i not in self.cube:
                raise KeyError(f"{i} not in {self.cube}")
        dictcoords = {}
        if len(reducted) <= 2:
            for coords in reducted:
                dictcoords[coords] = vars(self)[coords]
            axis = list(set(reducted) ^ set(self.cube))
            dictmesh: dict[str, Any] = {}
            # 2D map
            if len(axis) == 1:
                dictmesh[reducted[0]], dictmesh[reducted[1]] = np.meshgrid(
                    dictcoords[reducted[0]], dictcoords[reducted[1]]
                )
                axismed = "".join([axis[0], "med"])
                dictmesh[axis[0]] = vars(self)[axismed]
                # carefule: takes "xy", "yz", "zx" (all combinations)
                if "".join(reducted) in "".join((*self.cube, self.cube[0])):
                    ordered = True
                else:
                    ordered = False
                dictmesh["ordered"] = ordered
            # 1D curve
            else:
                dictmesh[reducted[0]] = vars(self)["".join([reducted[0], "med"])]
        else:
            raise ValueError(f"more than 2 coordinates were specified: {reducted}.")
        return dictmesh

    # on demande 'x','y' et la geometry est cartesian -> 'x','y'
    # on demande 'x','y' et la geometry est polaire -> 'R','phi'
    # on demande 'x','y' et la geometry est spherique -> 'r','phi'
    @overload
    def native_from_wanted(
        self, _wanted_x1: str, _wanted_x2: str, /
    ) -> tuple[tuple[str, str], str]: ...

    @overload
    def native_from_wanted(
        self, _wanted_x1: str, _wanted_x2: None, /
    ) -> tuple[tuple[str], str]: ...

    def native_from_wanted(self, _wanted_x1: str, _wanted_x2: Optional[str] = None, /):
        if self.geometry == "cartesian":
            conversion = {
                "x": "x",
                "y": "y",
                "z": "z",
            }
        elif self.geometry == "polar":
            conversion = {
                "R": "R",
                "phi": "phi",
                "z": "z",
                "x": "R",
                "y": "phi",
                "r": "R",
                "theta": "z",
            }
        elif self.geometry == "spherical":
            conversion = {
                "r": "r",
                "theta": "theta",
                "phi": "phi",
                "x": "r",
                "y": "phi",
                "z": "theta",
                "R": "r",
            }
        else:
            raise RuntimeError(f"Unknown geometry {self.geometry!r}")

        wanted: tuple[str, ...]
        if _wanted_x2 is None:
            wanted = (_wanted_x1,)
        else:
            wanted = (_wanted_x1, _wanted_x2)
        for i in wanted:
            if i not in conversion:
                raise KeyError(f"{i} not in {tuple(conversion.keys())}")
        if {"x", "y", "z"}.issuperset(wanted):
            target_geometry = "cartesian"
        elif {"R", "phi", "z"}.issuperset(wanted):
            target_geometry = "polar"
        elif {"r", "theta", "phi"}.issuperset(wanted):
            target_geometry = "spherical"
        else:
            raise ValueError(f"Unknown wanted plane: {wanted}.")

        native: tuple[str, ...]
        if _wanted_x2 is None:
            native = (conversion[_wanted_x1],)
        else:
            native = (conversion[_wanted_x1], conversion[_wanted_x2])

        return native, target_geometry

    # for 2D arrays
    def target_from_native(self, target_geometry, coords) -> dict[str, np.ndarray]:
        if self.geometry == "polar":
            R, phi, z = (coords["R"], coords["phi"], coords["z"])
            if target_geometry == "cartesian":
                x = R * np.cos(phi)
                y = R * np.sin(phi)
                target_coords = {"x": x, "y": y, "z": z}
            elif target_geometry == "spherical":
                r = np.sqrt(R**2 + z**2)
                theta = np.arctan2(R, z)
                target_coords = {"r": r, "theta": theta, "phi": phi}
            else:
                raise ValueError(f"Unknown target geometry {target_geometry}.")

        elif self.geometry == "cartesian":
            x, y, z = (coords["x"], coords["y"], coords["z"])
            if target_geometry == "polar":
                R = np.sqrt(x**2 + y**2)
                phi = np.arctan2(y, x)
                target_coords = {"R": R, "phi": phi, "z": z}
            elif target_geometry == "spherical":
                r = np.sqrt(x**2 + y**2 + z**2)
                theta = np.arctan2(np.sqrt(x**2 + y**2), z)
                phi = np.arctan2(y, x)
                target_coords = {"r": r, "theta": theta, "phi": phi}
            else:
                raise ValueError(f"Unknown target geometry {target_geometry}.")

        elif self.geometry == "spherical":
            r, theta, phi = (coords["r"], coords["theta"], coords["phi"])
            if target_geometry == "polar":
                R = r * np.sin(theta)
                z = r * np.cos(theta)
                target_coords = {"R": R, "phi": phi, "z": z}
            elif target_geometry == "cartesian":
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)
                if len(theta.shape) <= 1:
                    x = r * np.sin(theta) * np.cos(phi)
                    y = r * np.sin(theta) * np.sin(phi)
                    z = np.cos(theta)
                target_coords = {"x": x, "y": y, "z": z}
            else:
                raise ValueError(f"Unknown target geometry {target_geometry}.")

        else:
            raise ValueError(f"Unknown geometry {self.geometry}.")

        target_coords["ordered"] = coords["ordered"]
        return target_coords

    def _meshgrid_conversion(self, *wanted) -> dict:
        native_from_wanted = self.native_from_wanted(*wanted)
        native = native_from_wanted[0]
        target_geometry = native_from_wanted[1]
        native_meshcoords = self._meshgrid_reduction(*native)
        if len(wanted) == 1:
            return native_meshcoords
        else:
            meshcoords = {}
            if target_geometry == self.geometry:
                meshcoords = native_meshcoords
            else:
                meshcoords = self.target_from_native(target_geometry, native_meshcoords)
            return meshcoords


class GasField:
    """Idefix field class

    Attributes
    ==========
    data : array corresponding to the 3D data cube.
    coords : Coordinates of the data.
        Edge of the cells.
    coordsmed : Coordinates of the data.
        Center of the cells.
    """

    def __init__(
        self,
        field: str,
        data: np.ndarray,
        coords: Coordinates,
        ngeom: str,
        on: int,
        operation: str,
        *,
        inifile: Optional[PathT] = None,
        code: Union[str, Recipe, None] = None,
        directory: Optional[PathT] = None,
        rotate_by: Optional[float] = None,
        rotate_with: Optional[str] = None,
        rotate_grid: int = -1,  # deprecated
    ) -> None:
        self.field = field
        self.operation = operation
        self.native_geometry = ngeom
        self.data = data
        self.coords = coords
        self.on = on

        if directory is None:
            directory = Path.cwd()
        self.directory = Path(directory)

        self._loader = loader_from(
            code=code,
            parameter_file=inifile,
            directory=directory,
        )
        self.inifile = self._loader.parameter_file

        self._rotate_by = _parse_rotation_angle(
            rotate_by=rotate_by,
            rotate_with=rotate_with,
            planet_number_argument=(
                "rotate_grid",
                (rotate_grid if rotate_grid >= 0 else None),
            ),
            stacklevel=2,
            find_phip=self.find_phip,
        )

        # TODO: remove this after deprecation
        self._recipe = recipe_from(parameter_file=inifile, directory=directory)
        self._code = str(code or self._recipe)

    @property
    def code(self) -> str:
        warnings.warn(
            "GasField.code is deprecated and will be removed in a future version.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self._code

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        Returns
        =======
        shape : tuple
        """
        i, j, k = (max(1, n - 1) for n in self.coords.shape)
        return i, j, k

    def map(
        self,
        *wanted,
        rotate_by: Optional[float] = None,
        rotate_with: Optional[str] = None,
        planet_corotation: Optional[int] = None,  # deprecated
    ) -> Plotable:
        rotate_by = _parse_rotation_angle(
            rotate_by=rotate_by,
            rotate_with=rotate_with,
            planet_number_argument=("planet_corotation", planet_corotation),
            stacklevel=2,
            find_phip=self.find_phip,
        )

        data_key = self.field
        # we count the number of 1 in the shape of the data, which gives the real dimension of the data,
        # i.e. the number of reductions already performed (0 -> 3D, 1 -> 2D, 2 -> 1D)
        if self.shape.count(1) not in (1, 2):
            raise ValueError("data has to be 1D or 2D in order to call map.")
        dimension = len(wanted)

        meshgrid_conversion = self.coords._meshgrid_conversion(*wanted)

        if dimension == 1:
            abscissa_value = list(meshgrid_conversion.values())[0]
            abscissa_key = list(meshgrid_conversion.keys())[0]
            if "phi" in wanted and not _fequal(self._rotate_by, rotate_by):
                phicoord = self.coords.phi - rotate_by
                ipi = find_nearest(phicoord, 0)
                if abs(0 - phicoord[ipi]) > abs(
                    np.ediff1d(find_around(phicoord, 0))[0]
                ):
                    ipi = find_nearest(phicoord, 2 * np.pi)
                if self.native_geometry == "polar":
                    self.data = np.roll(self.data, -ipi + 1, axis=1)
                elif self.native_geometry == "spherical":
                    self.data = np.roll(self.data, -ipi + 1, axis=2)
                else:
                    raise NotImplementedError(
                        f"geometry flag '{self.native_geometry}' not implemented yet if corotation"
                    )
                self._rotate_by = rotate_by

            datamoved_tmp = np.moveaxis(self.data, self.shape.index(1), 0)
            datamoved = np.moveaxis(
                datamoved_tmp[0], datamoved_tmp[0].shape.index(1), 0
            )
            dict_plotable = {
                "abscissa": abscissa_key,
                "field": data_key,
                abscissa_key: abscissa_value,
                data_key: datamoved[0],
            }
        elif dimension == 2:
            # meshgrid in polar coordinates P, R (if "R", "phi") or R, P (if "phi", "R")
            # idem for all combinations of R,phi,z
            abscissa_value, ordinate_value = (
                meshgrid_conversion[wanted[0]],
                meshgrid_conversion[wanted[1]],
            )
            abscissa_key, ordinate_key = (wanted[0], wanted[1])
            native_from_wanted = self.coords.native_from_wanted(*wanted)[0]
            if "phi" in native_from_wanted and not _fequal(self._rotate_by, rotate_by):
                phicoord = self.coords.phi - rotate_by
                # ipi = find_nearest(phicoord, np.pi)
                # if (abs(np.pi-phicoord[ipi])>abs(np.ediff1d(find_around(phicoord, np.pi))[0])):
                #     ipi = find_nearest(phicoord, -np.pi)
                ipi = find_nearest(phicoord, 0)
                if abs(0 - phicoord[ipi]) > abs(
                    np.ediff1d(find_around(phicoord, 0))[0]
                ):
                    ipi = find_nearest(phicoord, 2 * np.pi)
                if self.native_geometry == "polar":
                    self.data = np.roll(self.data, -ipi + 1, axis=1)
                elif self.native_geometry == "spherical":
                    self.data = np.roll(self.data, -ipi + 1, axis=2)
                else:
                    raise NotImplementedError(
                        f"geometry flag '{self.native_geometry}' not implemented yet if corotation"
                    )
                self._rotate_by = rotate_by
            ordered = meshgrid_conversion["ordered"]
            # move the axis of reduction in the front in order to
            # perform the operation 3D(i,j,1) -> 2D(i,j) in a general way,
            # whatever the position of (i,j,1)
            # for that we must be careful to change the order ("ordered") if the data is reversed
            # in practice, this is tricky only if the "1" is in the middle:
            # 3D(i,1,k) -> 2D(i,k) is not a direct triedre anymore, we need to do 2D(i,k).T = 2D(k,i)
            position_of_3d_dimension = self.shape.index(1)
            datamoved = np.moveaxis(self.data, position_of_3d_dimension, 0)
            if position_of_3d_dimension == 1:
                ordered = not ordered
            if ordered:
                data_value = datamoved[0].T
            else:
                data_value = datamoved[0]

            dict_plotable = {
                "abscissa": abscissa_key,
                "ordinate": ordinate_key,
                "field": data_key,
                abscissa_key: abscissa_value,
                ordinate_key: ordinate_value,
                data_key: data_value,
            }
        else:
            raise RuntimeError

        assert dict_plotable[data_key].ndim == dimension

        return Plotable(dict_plotable)

    def save(
        self,
        directory: Optional[PathT] = None,
        header_only: bool = False,
    ) -> Path:
        if directory is None:
            directory = Path.cwd()
        else:
            directory = Path(directory)
        operation = self.operation or "_"
        headerdir = directory / "header"
        subdir = directory / self.field.lower()
        file = subdir / f"{operation}_{self.field}.{self.on:04d}.npy"
        if not header_only:
            if file.is_file():
                logger.info("{} already exists", file)
            else:
                subdir.mkdir(exist_ok=True, parents=True)
                with open(file, "wb") as fh:
                    np.save(fh, self.data)

        group_of_files = list(subdir.glob(f"{operation}*"))
        header_file = headerdir / f"header{operation}.json"
        if (len(group_of_files) > 0 and not header_file.is_file()) or header_only:
            headerdir.mkdir(exist_ok=True, parents=True)
            headername = headerdir / f"header{operation}.json"
            if headername.is_file():
                logger.info("{} already exists", headername)
            else:
                dictsaved = self.coords.get_attributes

                def is_array(item: tuple[str, Any]) -> bool:
                    _key, value = item
                    return isinstance(value, np.ndarray)

                for key, value in filter(is_array, dictsaved.items()):
                    dictsaved[key] = value.tolist()
                with open(headername, "w") as hfile:
                    json.dump(dictsaved, hfile, indent=2)

        src = self.inifile.resolve()
        dest = directory / self.inifile.name
        if dest != src:
            copyfile(src, dest)

        return file

    def find_ir(self, distance=1.0):
        if self.native_geometry in ("polar"):
            return find_nearest(self.coords.Rmed, distance)
        if self.native_geometry in ("spherical"):
            return find_nearest(self.coords.rmed, distance)

    def find_imid(self, altitude=0.0):
        if self.native_geometry in ("cartesian", "polar"):
            return find_nearest(self.coords.zmed, altitude)
        if self.native_geometry in ("spherical"):
            return find_nearest(self.coords.thetamed, np.pi / 2 - altitude)

    def find_iphi(self, phi=0):
        if self.native_geometry in ("polar", "spherical"):
            mod = len(self.coords.phi) - 1
            return find_nearest(self.coords.phi, phi) % mod

    def _load_planet(
        self,
        *,
        planet_number: Optional[int] = None,
        planet_file: Optional[str] = None,
    ) -> PlanetData:
        planet_file = _parse_planet_file(
            planet_number=planet_number, planet_file=planet_file
        )
        file = self.directory / planet_file
        return self._loader.load_planet_data(file)

    def _get_ind_output_number(self, time) -> int:
        ini = self._loader.load_ini_file()
        target_time = ini.output_time_interval * self.on
        return find_nearest(time, target_time)

    def find_rp(
        self,
        planet_number: Optional[int] = None,
        *,
        planet_file: Optional[str] = None,
    ) -> float:
        pd = self._load_planet(planet_number=planet_number, planet_file=planet_file)
        ind_on = self._get_ind_output_number(pd.t)
        return pd.d[ind_on]  # type: ignore [attr-defined]

    def find_rhill(
        self,
        planet_number: Optional[int] = None,
        *,
        planet_file: Optional[str] = None,
    ) -> float:
        ini = self._loader.load_ini_file()
        pd = self._load_planet(planet_number=planet_number, planet_file=planet_file)
        oe = pd.get_orbital_elements(ini.frame)
        ind_on = self._get_ind_output_number(pd.t)
        return pow(pd.q[ind_on] / 3.0, 1.0 / 3.0) * oe.a[ind_on]

    def find_phip(
        self,
        planet_number: Optional[int] = None,
        *,
        planet_file: Optional[str] = None,
    ) -> float:
        pd = self._load_planet(planet_number=planet_number, planet_file=planet_file)
        ind_on = self._get_ind_output_number(pd.t)
        return np.arctan2(pd.y, pd.x)[ind_on] % (2 * np.pi)

    def latitudinal_projection(self, theta=None, *, name_operation=None) -> "GasField":
        if name_operation is None:
            operation_suffix = "latitudinal_projection"
            if theta is not None:
                operation_suffix += str(np.pi/2-theta)
        else:
            operation_suffix = name_operation
        operation = "_".join([self.operation, operation_suffix])
        imid = self.find_imid()
        if self.native_geometry == "polar":
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.R,
                self.coords.phi,
                find_around(self.coords.z, self.coords.zmed[imid]),
            )
            # ret_coords = Coordinates(self.native_geometry, self.coords.R, find_around(self.coords.phi, self.coords.phimed[0]), self.coords.z)
            R = self.coords.Rmed
            z = self.coords.zmed
            integral = np.zeros((self.shape[0], self.shape[1]), dtype=">f4")
            # integral = np.zeros((self.shape[0],self.shape[2]), dtype='>f4')
            for i in range(self.shape[0]):
                km = find_nearest(z, z.min())
                kp = find_nearest(z, z.max())
                if theta is not None:
                    km = find_nearest(z, -R[i] * theta)
                    kp = find_nearest(z, R[i] * theta)
                integral[i, :] = np.sum(
                    (self.data[i, :, :] * np.ediff1d(self.coords.z)[None, :])[
                        :, km : kp + 1
                    ],
                    axis=1,
                    dtype="float64",
                )
                # integral[i,km] = -1
                # integral[i,kp] = 1
            ret_data = integral.reshape(self.shape[0], self.shape[1], 1)
            # ret_data = integral.reshape(self.shape[0],1,self.shape[2])
        if self.native_geometry == "spherical":
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.r,
                find_around(self.coords.theta, self.coords.thetamed[imid]),
                self.coords.phi,
            )
            km = find_nearest(self.coords.theta, self.coords.theta.min())
            kp = find_nearest(self.coords.theta, self.coords.theta.max())
            if theta is not None:
                km = find_nearest(self.coords.theta, np.pi / 2 + theta)
                kp = find_nearest(self.coords.theta, np.pi / 2 - theta)
            ret_data = (
                np.sum(
                    (
                        self.data
                        * self.coords.rmed[:, None, None]
                        * np.sin(self.coords.thetamed[None, :, None])
                        * np.ediff1d(self.coords.theta)[None, :, None]
                    )[:, km : kp + 1, :],
                    axis=1,
                    dtype="float64",
                )
            ).reshape(self.shape[0], 1, self.shape[2])
        return GasField(
            self.field,
            ret_data.astype("float32", copy=False),
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            directory=self.directory,
            rotate_by=self._rotate_by,
        )

    # def latitudinal_projection(self, theta=None):
    #     operation = self.operation + "_latitudinal_projection"
    #     imid = self.find_imid()
    #     if self.native_geometry == "polar":
    #         ret_coords = Coordinates(
    #             self.native_geometry,
    #             self.coords.R,
    #             self.coords.phi,
    #             find_around(self.coords.z, self.coords.zmed[imid]),
    #         )
    #         # ret_coords = Coordinates(self.native_geometry, self.coords.R, find_around(self.coords.phi, self.coords.phimed[0]), self.coords.z)
    #         R = self.coords.Rmed
    #         z = self.coords.zmed
    #         kall = []
    #         iall = []
    #         integral = np.zeros((self.shape[0], self.shape[1]), dtype=">f4")
    #         for i in range(self.shape[0]):
    #             km = find_nearest(z, z.min())
    #             kp = find_nearest(z, z.max())
    #             if theta is not None:
    #                 km = find_nearest(z, R[i] / np.tan(np.pi / 2 - theta))
    #                 kp = find_nearest(z, R[i] / np.tan(np.pi / 2 + theta))
    #             for k in range(kp, km - 2, -1):
    #                 kall.append(k)
    #                 iall.append(find_nearest(R, np.sqrt(R[i] ** 2 - z[k] ** 2)))
    #             knall = np.array(kall)[
    #                 np.where(
    #                     (z[kall] < R[iall] / np.tan(np.pi / 2 - theta))
    #                     & (z[kall] > R[iall] / np.tan(np.pi / 2 + theta))
    #                 )[0]
    #             ]
    #             inall = np.array(iall)[
    #                 np.where(
    #                     (z[kall] < R[iall] / np.tan(np.pi / 2 - theta))
    #                     & (z[kall] > R[iall] / np.tan(np.pi / 2 + theta))
    #                 )[0]
    #             ]
    #             kall = knall
    #             iall = inall
    #             integral[i, :] = np.nansum(
    #                 self.data[iall[:-1], :, kall[:-1]]
    #                 * R[iall[:-1]][:, None]
    #                 * np.ediff1d(np.arctan2(R[iall], z[kall]))[:, None],
    #                 axis=0,
    #                 dtype="float64",
    #             )
    #             iall = []
    #             kall = []
    #             # integral[i, :] = np.nansum(
    #             #     (self.data[i, :, :] * np.ediff1d(self.coords.z)[None, :])[
    #             #         :, km : kp + 1
    #             #     ],
    #             #     axis=1,
    #             #     dtype="float64",
    #             # )
    #             # integral[i,km] = -1
    #             # integral[i,kp] = 1
    #         ret_data = integral.reshape(self.shape[0], self.shape[1], 1)
    #         # ret_data = integral.reshape(self.shape[0],1,self.shape[2])
    #     if self.native_geometry == "spherical":
    #         ret_coords = Coordinates(
    #             self.native_geometry,
    #             self.coords.r,
    #             find_around(self.coords.theta, self.coords.thetamed[imid]),
    #             self.coords.phi,
    #         )
    #         km = find_nearest(self.coords.thetamed, self.coords.theta.min())
    #         kp = find_nearest(self.coords.thetamed, self.coords.theta.max())
    #         if theta is not None:
    #             km = find_nearest(self.coords.thetamed, np.pi / 2 - theta)
    #             kp = find_nearest(self.coords.thetamed, np.pi / 2 + theta)
    #         ret_data = (
    #             np.nansum(
    #                 (
    #                     self.data
    #                     * self.coords.rmed[:, None, None]
    #                     * np.sin(self.coords.thetamed[None, :, None])
    #                     * np.ediff1d(self.coords.theta)[None, :, None]
    #                 )[:, km : kp + 1, :],
    #                 axis=1,
    #                 dtype="float64",
    #             )
    #         ).reshape(self.shape[0], 1, self.shape[2])
    #     return GasField(
    #         self.field,
    #         np.float32(ret_data),
    #         ret_coords,
    #         self.native_geometry,
    #         self.on,
    #         operation,
    #         inifile=self.inifile,
    #         directory=self.directory,
    #         rotate_by=self._rotate_by,
    #     )

    def vertical_projection(self, z=None, *, name_operation=None) -> "GasField":
        if name_operation is None:
            if z is None:
                operation = self.operation + "_vertical_projection"
            else:
                operation = self.operation + f"_vertical_projection{z}"
        else:
            operation = self.operation + "_" + name_operation
        imid = self.find_imid()
        if self.native_geometry == "cartesian":
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.x,
                self.coords.y,
                find_around(self.coords.z, self.coords.zmed[imid]),
            )
            km = find_nearest(self.coords.zmed, self.coords.z.min())
            kp = find_nearest(self.coords.zmed, self.coords.z.max())
            if z is not None:
                km = find_nearest(self.coords.zmed, -z)
                kp = find_nearest(self.coords.zmed, z)
            ret_data = (
                np.nansum(
                    (self.data * np.ediff1d(self.coords.z))[:, :, km : kp + 1],
                    axis=2,
                    dtype="float64",
                )
            ).reshape(self.shape[0], self.shape[1], 1)
        if self.native_geometry == "polar":
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.R,
                self.coords.phi,
                find_around(self.coords.z, self.coords.zmed[imid]),
            )
            km = find_nearest(self.coords.zmed, self.coords.z.min())
            kp = find_nearest(self.coords.zmed, self.coords.z.max())
            if z is not None:
                km = find_nearest(self.coords.zmed, -z)
                kp = find_nearest(self.coords.zmed, z)
            ret_data = (
                np.nansum(
                    (self.data * np.ediff1d(self.coords.z))[:, :, km : kp + 1],
                    axis=2,
                    dtype="float64",
                )
            ).reshape(self.shape[0], self.shape[1], 1)
        if self.native_geometry == "spherical":
            raise NotImplementedError(
                """
                vertical_projection(z) function not implemented in spherical coordinates.\n
                Maybe you could use the function latitudinal_projection(theta)?
                """
            )
        return GasField(
            self.field,
            ret_data.astype("float32", copy=False),
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            directory=self.directory,
            rotate_by=self._rotate_by,
        )

    def vertical_at_midplane(self) -> "GasField":
        # self.field = r"%s$_{\rm mid}$" % self.field
        operation = self.operation + "_vertical_at_midplane"
        imid = self.find_imid()
        if self.native_geometry == "cartesian":
            # find_around looks around the 2 coords values that surround coordmed at imid
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.x,
                self.coords.y,
                find_around(self.coords.z, self.coords.zmed[imid]),
            )
            # ret_coords = Coordinates(self.native_geometry, self.coords.x, self.coords.y, self.coords.z[imid:imid+2])
            ret_data = self.data[:, :, imid].reshape(self.shape[0], self.shape[1], 1)
            # do geometry conversion!!! -> chainer la conversion (une fois que reduction de dimension -> conversion puis plot egalement chainable)
        if self.native_geometry == "polar":
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.R,
                self.coords.phi,
                find_around(self.coords.z, self.coords.zmed[imid]),
            )
            # ret_coords = Coordinates(self.native_geometry, self.coords.R, self.coords.phi, self.coords.z[imid:imid+2])
            ret_data = self.data[:, :, imid].reshape(self.shape[0], self.shape[1], 1)
        if self.native_geometry == "spherical":
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.r,
                find_around(self.coords.theta, self.coords.thetamed[imid]),
                self.coords.phi,
            )
            # ret_coords = Coordinates(self.native_geometry, self.coords.r, self.coords.theta[imid:imid+2], self.coords.phi)
            ret_data = self.data[:, imid, :].reshape(self.shape[0], 1, self.shape[2])
        return GasField(
            self.field,
            ret_data,
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            directory=self.directory,
            rotate_by=self._rotate_by,
        )

    def latitudinal_at_theta(self, theta=None, *, name_operation=None) -> "GasField":
        logger.info("latitudinal_at_theta TO BE TESTED")
        if theta is None:
            if self.native_geometry in ("cartesian", "polar", "spherical"):
                theta = 0.0
        if name_operation is None:
            operation = self.operation + f"_latitudinal_at_theta{np.pi/2-theta}"
        else:
            operation = self.operation + "_" + name_operation
        imid = self.find_imid(altitude=theta)
        if self.native_geometry == "polar":
            data_at_theta = np.zeros((self.shape[0], self.shape[1]), dtype=">f4")
            for i in range(self.shape[0]):
                if np.sign(theta) >= 0:
                    if (
                        find_nearest(
                            self.coords.zmed,
                            self.coords.R[i] / np.tan(np.pi / 2 - theta),
                        )
                        < self.shape[2]
                    ):
                        # print(i,find_nearest(rhoon.x,rhoon.x[i]),find_nearest(rhoon.z,4*0.05*rhoon.x[i]))
                        # rvz[find_nearest(rhoon.x,rhoon.x[i]),:,find_nearest(rhoon.z,4*0.05*rhoon.x[i])] = 1
                        data_at_theta[i, :] = self.data[
                            i,
                            :,
                            find_nearest(
                                self.coords.zmed,
                                self.coords.R[i] / np.tan(np.pi / 2 - theta),
                            ),
                        ]
                    else:
                        data_at_theta[i, :] = np.nan
                else:
                    if (
                        find_nearest(
                            self.coords.zmed,
                            self.coords.R[i] / np.tan(np.pi / 2 - theta),
                        )
                        > 0
                    ):
                        data_at_theta[i, :] = self.data[
                            i,
                            :,
                            find_nearest(
                                self.coords.zmed,
                                self.coords.R[i] / np.tan(np.pi / 2 - theta),
                            ),
                        ]
                    else:
                        data_at_theta[i, :] = np.nan
            ret_data = data_at_theta.reshape(self.shape[0], self.shape[1], 1)
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.R,
                self.coords.phi,
                find_around(self.coords.z, self.coords.zmed[imid]),
            )
        if self.native_geometry == "spherical":
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.r,
                find_around(self.coords.theta, self.coords.thetamed[imid]),
                self.coords.phi,
            )
            ret_data = self.data[
                :, find_nearest(self.coords.thetamed, np.pi / 2 - theta), :
            ].reshape(self.shape[0], 1, self.shape[2])
        return GasField(
            self.field,
            ret_data,
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            directory=self.directory,
            rotate_by=self._rotate_by,
        )

    def vertical_at_z(self, z=None, *, name_operation=None) -> "GasField":
        logger.info("vertical_at_z TO BE TESTED")
        # self.field = r"%s$_{\rm mid}$" % self.field
        if z is None:
            z = 0
        if name_operation is None:
            operation = self.operation + f"_vertical_at_z{z}"
        else:
            operation = self.operation + "_" + name_operation

        imid = self.find_imid(altitude=z)
        if self.native_geometry == "cartesian":
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.x,
                self.coords.y,
                find_around(self.coords.z, self.coords.zmed[imid]),
            )
            ret_data = self.data[:, :, find_nearest(self.coords.zmed, z)].reshape(
                self.shape[0], self.shape[1], 1
            )
        if self.native_geometry == "polar":
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.R,
                self.coords.phi,
                find_around(self.coords.z, self.coords.zmed[imid]),
            )
            ret_data = self.data[:, :, find_nearest(self.coords.zmed, z)].reshape(
                self.shape[0], self.shape[1], 1
            )
        if self.native_geometry == "spherical":
            raise NotImplementedError(
                "vertical at z in spherical coordinates not implemented yet."
            )
        """
            data_at_z = np.zeros((self.shape[0], self.shape[2]), dtype=">f4")
            for i in range(self.shape[0]):
                if np.sign(z) >= 0:
                    if (
                        find_nearest(
                            self.coords.thetamed, np.arctan2(self.coords.r[i], z)
                        )
                        < self.shape[2]
                    ):
                        # print(i,find_nearest(rhoon.x,rhoon.x[i]),find_nearest(rhoon.z,4*0.05*rhoon.x[i]))
                        # rvz[find_nearest(rhoon.x,rhoon.x[i]),:,find_nearest(rhoon.z,4*0.05*rhoon.x[i])] = 1
                        data_at_z[i, :] = self.data[
                            i,
                            find_nearest(
                                self.coords.thetamed, np.arctan2(self.coords.r[i], z)
                            ),
                            :,
                        ]
                    else:
                        data_at_z[i, :] = np.nan
                else:
                    if (
                        find_nearest(
                            self.coords.thetamed, np.arctan2(self.coords.r[i], z)
                        )
                        > 0
                    ):
                        data_at_z[i, :] = self.data[
                            i,
                            find_nearest(
                                self.coords.thetamed, np.arctan2(self.coords.r[i], z)
                            ),
                            :,
                        ]
                    else:
                        data_at_z[i, :] = np.nan
            ret_data = data_at_z.reshape(self.shape[0], 1, self.shape[2])
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.r,
                find_around(self.coords.theta, self.coords.thetamed[imid]),
                self.coords.phi,
            )
        """
        return GasField(
            self.field,
            ret_data,
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            directory=self.directory,
            rotate_by=self._rotate_by,
        )

    def azimuthal_at_phi(self, phi=None, *, name_operation=None) -> "GasField":
        if phi is None:
            phi = 0.0
        # self.field = r"%s ($\phi_P$)" % self.field
        if name_operation is None:
            operation = self.operation + f"_azimuthal_at_phi{phi}"
        else:
            operation = self.operation + "_" + name_operation
        # operation = self.operation + f"_azimuthal_at_phivortex"
        iphi = self.find_iphi(phi=phi)
        if self.native_geometry == "cartesian":
            raise NotImplementedError(
                f"geometry flag '{self.native_geometry}' not implemented yet for azimuthal_at_phi"
            )
        if self.native_geometry == "polar":
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.R,
                find_around(self.coords.phi, self.coords.phimed[iphi]),
                self.coords.z,
            )
            ret_data = self.data[:, iphi, :].reshape(self.shape[0], 1, self.shape[2])
        if self.native_geometry == "spherical":
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.r,
                self.coords.theta,
                find_around(self.coords.phi, self.coords.phimed[iphi]),
            )
            ret_data = self.data[:, :, iphi].reshape(self.shape[0], self.shape[1], 1)
        return GasField(
            self.field,
            ret_data,
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            directory=self.directory,
            rotate_by=self._rotate_by,
        )

    def azimuthal_at_planet(
        self, planet_number: Optional[int] = None, *, planet_file: Optional[str] = None
    ) -> "GasField":
        planet_file = _parse_planet_file(
            planet_number=planet_number, planet_file=planet_file
        )
        del planet_number
        operation = self.operation + "_azimuthal_at_planet"
        phip = self.find_phip(planet_file=planet_file)
        aziphip = self.azimuthal_at_phi(phi=phip)
        return GasField(
            self.field,
            aziphip.data,
            aziphip.coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            directory=self.directory,
            rotate_by=self._rotate_by,
        )

    def azimuthal_average(self) -> "GasField":
        # self.field = r"$\langle$%s$\rangle$" % self.field
        operation = self.operation + "_azimuthal_average"
        iphi = self.find_iphi(phi=0)
        if self.native_geometry == "cartesian":
            raise NotImplementedError(
                f"geometry flag '{self.native_geometry}' not implemented yet for azimuthal_average"
            )
        if self.native_geometry == "polar":
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.R,
                find_around(self.coords.phi, self.coords.phimed[iphi]),
                self.coords.z,
            )
            ret_data = np.nanmean(self.data, axis=1, dtype="float64").reshape(
                self.shape[0], 1, self.shape[2]
            )
        if self.native_geometry == "spherical":
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.r,
                self.coords.theta,
                find_around(self.coords.phi, self.coords.phimed[iphi]),
            )
            ret_data = np.nanmean(self.data, axis=2, dtype="float64").reshape(
                self.shape[0], self.shape[1], 1
            )
        return GasField(
            self.field,
            ret_data.astype("float32", copy=False),
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            directory=self.directory,
            rotate_by=self._rotate_by,
        )

    def remove_planet_hill(
        self, planet_number: Optional[int] = None, *, planet_file: Optional[str] = None
    ) -> "GasField":
        planet_file = _parse_planet_file(
            planet_number=planet_number, planet_file=planet_file
        )
        del planet_number
        # self.field = r"$\langle$%s$\rangle$" % self.field
        operation = self.operation + "_remove_planet_hill"
        phip = self.find_phip(planet_file=planet_file)
        rp = self.find_rp(planet_file=planet_file)
        rhill = self.find_rhill(planet_file=planet_file)
        iphip_m = self.find_iphi(phi=phip - 2 * rhill / rp)
        iphip_p = self.find_iphi(phi=phip + 2 * rhill / rp)
        if self.native_geometry == "cartesian":
            raise NotImplementedError(
                f"geometry flag '{self.native_geometry}' not implemented yet for azimuthal_average_except_planet_hill"
            )
        if self.native_geometry == "polar":
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.R,
                # find_around(self.coords.phi, phip),
                self.coords.phi,
                self.coords.z,
            )
            ret_data = self.data.copy()
            if iphip_p >= iphip_m and iphip_p != self.coords.shape[1]:
                ret_data[:, iphip_m : iphip_p + 1, :] = np.nan
            else:
                if iphip_p == self.coords.shape[1]:
                    ret_data[:, iphip_m:iphip_p, :] = np.nan
                else:
                    ret_data[:, 0 : iphip_p + 1, :] = np.nan
                    ret_data[:, iphip_m : self.coords.shape[1], :] = np.nan
            # ret_data = np.nanmean(self.data, axis=1, dtype="float64").reshape(
            #     self.shape[0], 1, self.shape[2]
            # )
        if self.native_geometry == "spherical":
            ret_coords = Coordinates(
                self.native_geometry,
                self.coords.r,
                self.coords.theta,
                # find_around(self.coords.phi, phip),
                self.coords.phi,
            )
            ret_data = self.data.copy()
            if iphip_p >= iphip_m and iphip_p != self.coords.shape[2]:
                ret_data[:, :, iphip_m : iphip_p + 1] = np.nan
            else:
                if iphip_p == self.coords.shape[2]:
                    ret_data[:, :, iphip_m:iphip_p] = np.nan
                else:
                    ret_data[:, :, 0 : iphip_p + 1] = np.nan
                    ret_data[:, :, iphip_m : self.coords.shape[2]] = np.nan
            # ret_data = np.nanmean(self.data, axis=2, dtype="float64").reshape(
            #     self.shape[0], self.shape[1], 1
            # )
        return GasField(
            self.field,
            ret_data,
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            directory=self.directory,
            rotate_by=self._rotate_by,
        )

    def radial_at_r(self, distance=None, *, name_operation=None) -> "GasField":
        if distance is None:
            distance = 1.0
        if name_operation is None:
            operation = self.operation + f"_radial_at_r{distance}"
        else:
            operation = self.operation + "_" + name_operation
        ir1 = self.find_ir(distance=distance)
        if self.native_geometry == "cartesian":
            raise NotImplementedError(
                f"geometry flag '{self.native_geometry}' not implemented yet for radial_at_r"
            )
        if self.native_geometry == "polar":
            # self.field = r"%s (R=1)" % self.field
            ret_coords = Coordinates(
                self.native_geometry,
                find_around(self.coords.R, self.coords.Rmed[ir1]),
                self.coords.phi,
                self.coords.z,
            )
        if self.native_geometry == "spherical":
            # self.field = r"%s (r=1)" % self.field
            ret_coords = Coordinates(
                self.native_geometry,
                find_around(self.coords.r, self.coords.rmed[ir1]),
                self.coords.theta,
                self.coords.phi,
            )
        ret_data = self.data[ir1, :, :].reshape(1, self.shape[1], self.shape[2])
        return GasField(
            self.field,
            ret_data,
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            directory=self.directory,
            rotate_by=self._rotate_by,
        )

    def radial_average_interval(
        self, vmin=None, vmax=None, *, name_operation=None
    ) -> "GasField":
        if (vmin is None) or (vmax is None):
            raise ValueError(
                f"The radial interval {vmin=} and {vmax=} should be defined"
            )
        if name_operation is None:
            operation = self.operation + f"_radial_average_interval_{vmin}_{vmax}"
        else:
            operation = self.operation + "_" + name_operation
        irmin = self.find_ir(distance=vmin)
        irmax = self.find_ir(distance=vmax)
        ir = self.find_ir(distance=(vmax - vmin) / 2)
        if self.native_geometry == "cartesian":
            raise NotImplementedError(
                f"geometry flag '{self.native_geometry}' not implemented yet for radial_at_r"
            )
        if self.native_geometry == "polar":
            if vmin is None:
                vmin = self.coords.R.min()
            if vmax is None:
                vmax = self.coords.R.max()
            ret_coords = Coordinates(
                self.native_geometry,
                find_around(self.coords.R, self.coords.Rmed[ir]),
                self.coords.phi,
                self.coords.z,
            )
        if self.native_geometry == "spherical":
            if vmin is None:
                vmin = self.coords.r.min()
            if vmax is None:
                vmax = self.coords.r.max()
            ret_coords = Coordinates(
                self.native_geometry,
                find_around(self.coords.r, self.coords.rmed[ir]),
                self.coords.theta,
                self.coords.phi,
            )
        ret_data = np.nanmean(
            self.data[irmin : irmax + 1, :, :], axis=0, dtype="float64"
        ).reshape(1, self.shape[1], self.shape[2])
        return GasField(
            self.field,
            ret_data,
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            directory=self.directory,
            rotate_by=self._rotate_by,
        )

    def diff(self, on_2) -> "GasField":
        ds_2 = GasDataSet(
            on_2,
            geometry=self.native_geometry,
            inifile=self.inifile,
            directory=self.directory,
        )
        if self.operation != "":
            raise KeyError(
                "For now, diff should only be applied on the initial Field cube."
            )
        # self.operation += "diff"
        ret_data = (self.data - ds_2[self.field].data) / ds_2[self.field].data
        ret_coords = self.coords
        # self.field = r"$\frac{%s - %s_0}{%s_0}$" % (self.field, self.field, self.field)
        return GasField(
            self.field,
            ret_data,
            ret_coords,
            self.native_geometry,
            self.on,
            self.operation,
            inifile=self.inifile,
            directory=self.directory,
            rotate_by=self._rotate_by,
        )

    def rotate(
        self,
        planet_corotation: Optional[int] = None,
        *,
        rotate_with: Optional[str] = None,
        rotate_by: Optional[float] = None,
    ) -> "GasField":
        rotate_by = _parse_rotation_angle(
            rotate_by=rotate_by,
            rotate_with=rotate_with,
            planet_number_argument=("planet_corotation", planet_corotation),
            find_phip=self.find_phip,
            stacklevel=2,
        )

        operation = self.operation
        if self.shape.count(1) > 1:
            raise ValueError("data has to be 2D or 3D in order to rotate the data.")
        if not _fequal(self._rotate_by, rotate_by):
            phicoord = self.coords.phi - rotate_by
            ipi = find_nearest(phicoord, 0)
            if abs(0 - phicoord[ipi]) > abs(np.ediff1d(find_around(phicoord, 0))[0]):
                ipi = find_nearest(phicoord, 2 * np.pi)
            if self.native_geometry == "polar":
                ret_data = np.roll(
                    self.data,
                    -ipi + 1,
                    axis=1,
                )
                ret_coords = Coordinates(
                    self.native_geometry, self.coords.R, self.coords.phi, self.coords.z
                )
            elif self.native_geometry == "spherical":
                ret_data = np.roll(
                    self.data,
                    -ipi + 1,
                    axis=2,
                )
                ret_coords = Coordinates(
                    self.native_geometry,
                    self.coords.r,
                    self.coords.theta,
                    self.coords.phi,
                )
            else:
                raise NotImplementedError(
                    f"geometry flag '{self.native_geometry}' not implemented yet if corotation"
                )
            self._rotate_by = rotate_by
        else:
            ret_data = self.data
            if self.native_geometry == "polar":
                ret_coords = Coordinates(
                    self.native_geometry, self.coords.R, self.coords.phi, self.coords.z
                )
            elif self.native_geometry == "spherical":
                ret_coords = Coordinates(
                    self.native_geometry,
                    self.coords.r,
                    self.coords.theta,
                    self.coords.phi,
                )

        return GasField(
            self.field,
            ret_data,
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            directory=self.directory,
            rotate_by=self._rotate_by,
        )


class GasDataSet:
    """Idefix dataset class that contains everything in the .vtk file

    Args:
        input_dataset (int or str): output number or file name
        directory (str): directory of the .vtk
        geometry (str): for retrocompatibility if old vtk format
        inifile (str): name of the simulation's parameter file if no default files (combined with code)
        code (str): name of the code ("idefix", "pluto", "fargo3d", "fargo-adsg")
    Returns:
        dataset
    """

    def __init__(
        self,
        input_dataset: Union[int, PathT],
        /,
        *,
        inifile: Optional[PathT] = None,
        code: Union[str, Recipe, None] = None,
        geometry: Optional[str] = None,
        directory: Optional[PathT] = None,
        fluid: Optional[str] = None,
        operation: Optional[str] = None,
    ) -> None:
        if isinstance(input_dataset, (str, Path)):
            input_dataset = Path(input_dataset)
            directory_from_input = input_dataset.parent
            if directory is None:
                directory = directory_from_input
            elif directory_from_input.resolve() != Path(directory).resolve():
                raise ValueError(
                    f"directory value {directory!r} does not match "
                    f"directory name from input_dataset ({directory_from_input!r})"
                )
            del directory_from_input

        if directory is None:
            directory = Path.cwd()
        else:
            directory = Path(directory)

        recipe = recipe_from(
            code=code,
            parameter_file=inifile,
            directory=directory,
        )

        if fluid is not None and recipe is not Recipe.FARGO3D:
            warnings.warn(
                "Unused keyword argument: 'fluid'",
                category=UserWarning,
                stacklevel=2,
            )

        loader = loader_from(
            code=code,
            parameter_file=inifile,
            directory=directory,
        )
        if operation is not None:
            ignored_kwargs = []
            msg = ""
            if fluid is not None:
                ignored_kwargs.append("fluid")
            if geometry is not None:
                ignored_kwargs.append("geometry")
            if ignored_kwargs:
                ignored = ", ".join(repr(_) for _ in ignored_kwargs)
                msg = (
                    "The following keyword arguments are ignored "
                    f"when combined with 'operation': {ignored}"
                )
                warnings.warn(msg, UserWarning, stacklevel=2)
            self._loader = dataclasses.replace(loader, binary_reader=NPYReader)
        else:
            self._loader = loader

        self.on, datafile = self._loader.binary_reader.parse_output_number_and_filename(
            input_dataset,
            directory=directory,
            prefix=operation or "",
        )

        self._read = self._loader.load_bin_data(
            datafile,
            geometry=geometry,
            fluid=fluid,
        )

        self.native_geometry = self._read.geometry
        self.dict = self._read.data
        self.coords = Coordinates(
            self.native_geometry,
            self._read.x1,
            self._read.x2,
            self._read.x3,
        )
        for key in self.dict:
            self.dict[key] = GasField(
                key,
                self.dict[key],
                self.coords,
                self.native_geometry,
                self.on,
                operation="",
                inifile=self._loader.parameter_file,
                code=recipe,
                directory=directory,
            )

        # backward compatibility for self.params
        self._parameters_input = {
            "inifile": inifile,
            "code": code.removesuffix("_vtk") if code is not None else None,
            "directory": directory,
        }

    @cached_property
    def params(self):
        from nonos.api.from_simulation import Parameters

        warnings.warn(
            "GasDataSet.params is deprecated "
            "and will be removed in a future version.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return Parameters(**self._parameters_input)

    @classmethod
    def from_npy(
        cls,
        on: int,
        *,
        inifile: Optional[PathT] = None,
        code: Union[str, Recipe, None] = None,
        directory: Optional[PathT] = None,
        operation: str,
    ) -> "GasDataSet":
        warnings.warn(
            "GasDataSet.from_npy is deprecated "
            "and will be removed in a future version. "
            "Instead, call GasDataSet(...) directly.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return GasDataSet(
            on,
            inifile=inifile,
            code=code,
            directory=directory,
            operation=operation,
        )

    def __getitem__(self, key) -> "GasField":
        if key in self.dict:
            return self.dict[key]
        else:
            raise KeyError

    def keys(self) -> KeysView[str]:
        """
        Returns
        =======
        keys of the dict
        """
        return self.dict.keys()

    def values(self) -> ValuesView["GasField"]:
        """
        Returns
        =======
        values of the dict
        """
        return self.dict.values()

    def items(self) -> ItemsView[str, "GasField"]:
        """
        Returns
        =======
        items of the dict
        """
        return self.dict.items()

    @property
    def nfields(self) -> int:
        """
        Returns
        =======
        The number of fields in the GasDataSet
        """
        return len(self.dict)
