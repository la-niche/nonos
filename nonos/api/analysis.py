import glob
import os
import warnings
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from matplotlib.ticker import SymmetricalLogLocator

from nonos.api.from_simulation import Parameters
from nonos.api.tools import find_around, find_nearest
from nonos.logging import logger


class Plotable:
    def __init__(self, dict_plotable: dict):
        self.dict_plotable = dict_plotable
        self.data = self.dict_plotable[self.dict_plotable["field"]]
        self.dimension = len(self.data.shape)

    def plot(
        self,
        fig,
        ax,
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
                    cb_axis.set_major_locator(SymmetricalLogLocator(trf))
                    if float(trf.base).is_integer():
                        locator = SymmetricalLogLocator(
                            trf, subs=np.arange(1, trf.base)
                        )
                        cb_axis.set_minor_locator(locator)
            else:
                return im
        if self.dimension == 1:
            vmin = kwargs.pop("vmin") if "vmin" in kwargs else np.nanmin(data)
            vmax = kwargs.pop("vmax") if "vmax" in kwargs else np.nanmax(data)
            self.akey = self.dict_plotable["abscissa"]
            self.avalue = self.dict_plotable[self.akey]
            if "norm" in kwargs:
                logger.info("norm has no meaning in 1D.")
                kwargs.pop("norm")
            im = ax.plot(
                self.avalue,
                data,
                **kwargs,
            )
            ax.set_ylim(ymin=vmin)
            ax.set_ylim(ymax=vmax)
            ax.set_xlabel(self.akey)
            if title is not None:
                ax.set_ylabel(title)
        if filename is not None:
            fig.savefig(f"{filename}.{fmt}", bbox_inches="tight", dpi=dpi)


class Coordinates:
    """Coordinates class from x1, x2, x3"""

    def __init__(self, geometry: str, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray):
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
            if self.phi.max() - np.pi > np.pi / 2:
                self.phi -= np.pi
            self.cube = ("r", "theta", "phi")
            self.rmed = 0.5 * (self.r[1:] + self.r[:-1])
            self.thetamed = 0.5 * (self.theta[1:] + self.theta[:-1])
            self.phimed = 0.5 * (self.phi[1:] + self.phi[:-1])
        if self.geometry == "polar":
            self.R = x1
            self.phi = x2
            if self.phi.max() - np.pi > np.pi / 2:
                self.phi -= np.pi
            self.z = x3
            self.cube = ("R", "phi", "z")
            self.Rmed = 0.5 * (self.R[1:] + self.R[:-1])
            self.phimed = 0.5 * (self.phi[1:] + self.phi[:-1])
            self.zmed = 0.5 * (self.z[1:] + self.z[:-1])

    @property
    def shape(self):
        """
        Returns
        =======
        shape : tuple
        """
        if self.geometry == "cartesian":
            return len(self.x), len(self.y), len(self.z)
        if self.geometry == "spherical":
            return len(self.r), len(self.theta), len(self.phi)
        if self.geometry == "polar":
            return len(self.R), len(self.phi), len(self.z)

    @property
    def get_attributes(self):
        if self.geometry == "cartesian":
            return {"geometry": self.geometry, "x": self.x, "y": self.y, "z": self.z}
        if self.geometry == "spherical":
            return {
                "geometry": self.geometry,
                "r": self.r,
                "theta": self.theta,
                "phi": self.phi,
            }
        if self.geometry == "polar":
            return {
                "geometry": self.geometry,
                "R": self.R,
                "phi": self.phi,
                "z": self.z,
            }

    @property
    def get_coords(self):
        if self.geometry == "cartesian":
            return {
                "x": self.x,
                "y": self.y,
                "z": self.z,
                "xmed": self.xmed,
                "ymed": self.ymed,
                "zmed": self.zmed,
            }
        if self.geometry == "spherical":
            return {
                "r": self.r,
                "theta": self.theta,
                "phi": self.phi,
                "rmed": self.rmed,
                "thetamed": self.thetamed,
                "phimed": self.phimed,
            }
        if self.geometry == "polar":
            return {
                "R": self.R,
                "phi": self.phi,
                "z": self.z,
                "Rmed": self.Rmed,
                "phimed": self.phimed,
                "zmed": self.zmed,
            }

    def _meshgrid_reduction(self, *reducted):
        for i in reducted:
            if i not in self.cube:
                raise KeyError(f"{i} not in {self.cube}")
        dictcoords = {}
        if len(reducted) <= 2:
            for coords in reducted:
                dictcoords[coords] = vars(self)[coords]
            axis = list(set(reducted) ^ set(self.cube))
            dictmesh = {}
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
    def native_from_wanted(self, *wanted):
        if self.geometry == "cartesian":
            conversion = {
                "x": "x",
                "y": "y",
                "z": "z",
            }
        if self.geometry == "polar":
            conversion = {
                "R": "R",
                "phi": "phi",
                "z": "z",
                "x": "R",
                "y": "phi",
                "r": "R",
                "theta": "z",
            }
        if self.geometry == "spherical":
            conversion = {
                "r": "r",
                "theta": "theta",
                "phi": "phi",
                "x": "r",
                "y": "phi",
                "z": "theta",
                "R": "r",
            }
        for i in wanted:
            if i not in tuple(conversion.keys()):
                raise KeyError(f"{i} not in {tuple(conversion.keys())}")
        if set(wanted) & {"x", "y", "z"} == set(wanted):
            target_geometry = "cartesian"
        elif set(wanted) & {"R", "phi", "z"} == set(wanted):
            target_geometry = "polar"
        elif set(wanted) & {"r", "theta", "phi"} == set(wanted):
            target_geometry = "spherical"
        else:
            raise ValueError(f"Unknown wanted plane: {wanted}.")
        native = tuple(conversion[i] for i in wanted)
        return native, target_geometry

    # for 2D arrays
    def target_from_native(self, target_geometry, coords):
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

    def _meshgrid_conversion(self, *wanted):
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
        inifile: str = "",
        code: str = "",
        directory="",
        rotate_grid: int = -1,
    ):
        self.field = field
        self.operation = operation
        self.native_geometry = ngeom
        self.data = data
        self.coords = coords
        self.on = on

        self.inifile = inifile
        self.code = code
        self.directory = directory
        self._rotate_grid = rotate_grid

    @property
    def shape(self) -> Tuple[Any, ...]:
        """
        Returns
        =======
        shape : tuple
        """
        return tuple(i - 1 if i > 1 else i for i in self.coords.shape)

    def map(self, *wanted, planet_corotation: Optional[int] = None):
        data_key = self.field
        # we count the number of 1 in the shape of the data, which gives the real dimension of the data,
        # i.e. the number of reductions already performed (0 -> 3D, 1 -> 2D, 2 -> 1D)
        if self.shape.count(1) not in (1, 2):
            raise ValueError("data has to be 1D or 2D in order to call map.")
        dimension = len(wanted)

        if dimension == 1:
            meshgrid_conversion = self.coords._meshgrid_conversion(*wanted)
            # abscissa = meshgrid_conversion[wanted[0]]
            abscissa_value = list(meshgrid_conversion.values())[0]
            abscissa_key = list(meshgrid_conversion.keys())[0]
            if (
                planet_corotation is not None
                and "phi" in wanted
                and self._rotate_grid != planet_corotation
            ):
                phip = self.find_phip(planet_number=planet_corotation)
                phicoord = self.coords.phi - phip  # - np.pi
                ipi = find_nearest(phicoord, 0)
                if abs(0 - phicoord[ipi]) > abs(
                    np.ediff1d(find_around(phicoord, 0))[0]
                ):
                    ipi = find_nearest(phicoord, 2 * np.pi)
                if self.native_geometry == "polar":
                    self.data = np.roll(
                        self.data, -ipi - self.coords.phi.shape[0] // 2 + 1, axis=1
                    )
                elif self.native_geometry == "spherical":
                    self.data = np.roll(
                        self.data, -ipi - self.coords.phi.shape[0] // 2 + 1, axis=2
                    )
                else:
                    raise NotImplementedError(
                        f"geometry flag '{self.native_geometry}' not implemented yet if planet_corotation"
                    )

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
        if dimension == 2:
            # meshgrid in polar coordinates P, R (if "R", "phi") or R, P (if "phi", "R")
            # idem for all combinations of R,phi,z
            meshgrid_conversion = self.coords._meshgrid_conversion(*wanted)
            abscissa_value, ordinate_value = (
                meshgrid_conversion[wanted[0]],
                meshgrid_conversion[wanted[1]],
            )
            abscissa_key, ordinate_key = (wanted[0], wanted[1])
            native_from_wanted = self.coords.native_from_wanted(*wanted)[0]
            if (
                planet_corotation is not None
                and "phi" in native_from_wanted
                and self._rotate_grid != planet_corotation
            ):
                phip = self.find_phip(planet_number=planet_corotation)
                phicoord = self.coords.phi - phip  # - np.pi
                # ipi = find_nearest(phicoord, np.pi)
                # if (abs(np.pi-phicoord[ipi])>abs(np.ediff1d(find_around(phicoord, np.pi))[0])):
                #     ipi = find_nearest(phicoord, -np.pi)
                ipi = find_nearest(phicoord, 0)
                if abs(0 - phicoord[ipi]) > abs(
                    np.ediff1d(find_around(phicoord, 0))[0]
                ):
                    ipi = find_nearest(phicoord, 2 * np.pi)
                if self.native_geometry == "polar":
                    self.data = np.roll(
                        self.data, -ipi - self.coords.phi.shape[0] // 2 + 1, axis=1
                    )
                elif self.native_geometry == "spherical":
                    self.data = np.roll(
                        self.data, -ipi - self.coords.phi.shape[0] // 2 + 1, axis=2
                    )
                else:
                    raise NotImplementedError(
                        f"geometry flag '{self.native_geometry}' not implemented yet if planet_corotation"
                    )
                self._rotate_grid = planet_corotation
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
        return Plotable(dict_plotable)

    def save(self, directory="", header_only=False):
        if not header_only:
            if not os.path.exists(os.path.join(directory, self.field.lower())):
                os.mkdir(os.path.join(directory, self.field.lower()))
            filename = os.path.join(
                directory,
                self.field.lower(),
                f"{self.operation}_{self.field}.{self.on:04d}.npy",
            )
            if Path(filename).is_file():
                logger.info("{} already exists", filename)
            else:
                with open(filename, "wb") as file:
                    np.save(file, self.data)

        group_of_files = list(
            glob.glob1(
                os.path.join(directory, self.field.lower()), f"{self.operation}*"
            )
        )
        header_file = list(
            glob.glob1(os.path.join(directory, "header"), f"header{self.operation}.npy")
        )
        if (len(group_of_files) > 0 and len(header_file) == 0) or header_only:
            if not os.path.exists(os.path.join(directory, "header")):
                os.mkdir(os.path.join(directory, "header"))
            headername = os.path.join(
                directory, "header", f"header{self.operation}.npy"
            )
            if Path(headername).is_file():
                logger.info("{} already exists", headername)
            else:
                dictsaved = self.coords.get_attributes
                with open(headername, "wb") as file:
                    np.save(file, dictsaved)

    def find_ir(self, distance=1.0):
        r1 = distance
        if self.native_geometry in ("polar"):
            return find_nearest(self.coords.Rmed, r1)
        if self.native_geometry in ("spherical"):
            return find_nearest(self.coords.rmed, r1)

    def find_imid(self, altitude=None):
        if altitude is None:
            if self.native_geometry in ("cartesian", "polar", "spherical"):
                altitude = 0.0
        if self.native_geometry in ("cartesian", "polar"):
            return find_nearest(self.coords.zmed, altitude)
        if self.native_geometry in ("spherical"):
            return find_nearest(self.coords.thetamed, np.pi / 2 - altitude)

    def find_iphi(self, phi=0):
        if self.native_geometry in ("polar", "spherical"):
            return find_nearest(self.coords.phi, phi) % self.coords.phimed.shape[0]

    def find_rp(self, planet_number: int = 0):
        init = Parameters(
            inifile=self.inifile, code=self.code, directory=self.directory
        )
        init.loadIniFile()
        init.loadPlanetFile(planet_number=planet_number)
        ind_on = find_nearest(init.tpl, init.vtk * self.on)
        return init.dpl[ind_on]

    def find_rhill(self, planet_number: int = 0):
        init = Parameters(
            inifile=self.inifile, code=self.code, directory=self.directory
        )
        init.loadIniFile()
        init.loadPlanetFile(planet_number=planet_number)
        ind_on = find_nearest(init.tpl, init.vtk * self.on)
        return pow(init.qpl[ind_on] / 3.0, 1.0 / 3.0) * init.apl[ind_on]

    def find_phip(self, planet_number: int = 0):
        if self._rotate_grid == planet_number:
            return 0.0
        init = Parameters(
            inifile=self.inifile, code=self.code, directory=self.directory
        )
        init.loadIniFile()
        init.loadPlanetFile(planet_number=planet_number)
        ind_on = find_nearest(init.tpl, init.vtk * self.on)
        return np.arctan2(init.ypl, init.xpl)[ind_on] % (2 * np.pi) - np.pi

    def latitudinal_projection(self, theta=None):
        operation = self.operation + "_latitudinal_projection"
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
            np.float32(ret_data),
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            code=self.code,
            directory=self.directory,
            rotate_grid=self._rotate_grid,
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
    #         code=self.code,
    #         directory=self.directory,
    #         rotate_grid=self._rotate_grid,
    #     )

    def vertical_projection(self, z=None):
        operation = self.operation + "_vertical_projection"
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
            np.float32(ret_data),
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            code=self.code,
            directory=self.directory,
            rotate_grid=self._rotate_grid,
        )

    def vertical_at_midplane(self):
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
            code=self.code,
            directory=self.directory,
            rotate_grid=self._rotate_grid,
        )

    def latitudinal_at_theta(self, theta=None, name_operation=None):
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
            code=self.code,
            directory=self.directory,
            rotate_grid=self._rotate_grid,
        )

    def vertical_at_z(self, z=None, name_operation=None):
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
        return GasField(
            self.field,
            ret_data,
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            code=self.code,
            directory=self.directory,
            rotate_grid=self._rotate_grid,
        )

    def azimuthal_at_phi(self, phi=None):
        if phi is None:
            phi = 0.0
        # self.field = r"%s ($\phi_P$)" % self.field
        operation = self.operation + f"_azimuthal_at_phi{phi}"
        # operation = self.operation + f"_azimuthal_at_phivortex"
        iphi = self.find_iphi(phi=phi)
        if self.native_geometry == "cartesian":
            raise NotImplementedError(
                f"geometry flag '{self._native_geometry}' not implemented yet for azimuthal_at_phi"
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
            code=self.code,
            directory=self.directory,
            rotate_grid=self._rotate_grid,
        )

    def azimuthal_at_planet(self, planet_number: int = 0):
        operation = self.operation + "_azimuthal_at_planet"
        phip = self.find_phip(planet_number=planet_number)
        aziphip = self.azimuthal_at_phi(phi=phip)
        return GasField(
            self.field,
            aziphip.data,
            aziphip.coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            code=self.code,
            directory=self.directory,
            rotate_grid=self._rotate_grid,
        )

    def azimuthal_average(self):
        # self.field = r"$\langle$%s$\rangle$" % self.field
        operation = self.operation + "_azimuthal_average"
        iphi = self.find_iphi(phi=0)
        if self.native_geometry == "cartesian":
            raise NotImplementedError(
                f"geometry flag '{self._native_geometry}' not implemented yet for azimuthal_average"
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
            np.float32(ret_data),
            ret_coords,
            self.native_geometry,
            self.on,
            operation,
            inifile=self.inifile,
            code=self.code,
            directory=self.directory,
            rotate_grid=self._rotate_grid,
        )

    def remove_planet_hill(self, planet_number: int = 0):
        # self.field = r"$\langle$%s$\rangle$" % self.field
        operation = self.operation + "_remove_planet_hill"
        phip = self.find_phip(planet_number=planet_number)
        rp = self.find_rp(planet_number=planet_number)
        rhill = self.find_rhill(planet_number=planet_number)
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
            code=self.code,
            directory=self.directory,
            rotate_grid=self._rotate_grid,
        )

    def radial_at_r(self, distance=None):
        if distance is None:
            distance = 1.0
        operation = self.operation + f"_radial_at_r{distance}"
        ir1 = self.find_ir(distance=distance)
        if self.native_geometry == "cartesian":
            raise NotImplementedError(
                f"geometry flag '{self._native_geometry}' not implemented yet for radial_at_r"
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
            code=self.code,
            directory=self.directory,
            rotate_grid=self._rotate_grid,
        )

    def radial_average_interval(self, vmin=None, vmax=None):
        operation = self.operation + f"_radial_average_interval_{vmin}_{vmax}"
        irmin = self.find_ir(distance=vmin)
        irmax = self.find_ir(distance=vmax)
        ir = self.find_ir(distance=(vmax - vmin) / 2)
        if self.native_geometry == "cartesian":
            raise NotImplementedError(
                f"geometry flag '{self._native_geometry}' not implemented yet for radial_at_r"
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
            code=self.code,
            directory=self.directory,
            rotate_grid=self._rotate_grid,
        )

    def diff(self, on_2):
        ds_2 = GasDataSet(
            on_2,
            geometry=self.native_geometry,
            inifile=self.inifile,
            code=self.code,
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
            code=self.code,
            directory=self.directory,
            rotate_grid=self._rotate_grid,
        )

    def rotate(self, planet_corotation: Optional[int] = None):
        operation = self.operation
        if self.shape.count(1) != 1:
            raise ValueError("data has to be 2D in order to rotate the data.")
        if (
            planet_corotation is not None and self._rotate_grid != planet_corotation
        ):  # and self.coords.phi.shape[0]!=1:
            phip = self.find_phip(planet_number=planet_corotation)
            phicoord = self.coords.phi - phip  # - np.pi
            ipi = find_nearest(phicoord, 0)
            if abs(0 - phicoord[ipi]) > abs(np.ediff1d(find_around(phicoord, 0))[0]):
                ipi = find_nearest(phicoord, 2 * np.pi)
            if self.native_geometry == "polar":
                ret_data = np.roll(
                    self.data, -ipi - self.coords.phi.shape[0] // 2 + 1, axis=1
                )
                ret_coords = Coordinates(
                    self.native_geometry, self.coords.R, self.coords.phi, self.coords.z
                )
            elif self.native_geometry == "spherical":
                ret_data = np.roll(
                    self.data, -ipi - self.coords.phi.shape[0] // 2 + 1, axis=2
                )
                ret_coords = Coordinates(
                    self.native_geometry,
                    self.coords.r,
                    self.coords.theta,
                    self.coords.phi,
                )
            else:
                raise NotImplementedError(
                    f"geometry flag '{self.native_geometry}' not implemented yet if planet_corotation"
                )
            self._rotate_grid = planet_corotation
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
            code=self.code,
            directory=self.directory,
            rotate_grid=self._rotate_grid,
        )


class GasDataSet:
    """Idefix dataset class that contains everything in the .vtk file

    Args:
        on (int): output number
        directory (str): directory of the .vtk
        geometry (str): for retrocompatibility if old vtk format
        inifile (str): name of the simulation's parameter file if no default files (combined with code)
        code (str): name of the code ("idefix", "pluto", "fargo3d", "fargo-adsg")
    Returns:
        dataset
    """

    def __init__(
        self,
        on: int,
        *,
        inifile: str = "",
        code: str = "",
        geometry: str = "unknown",
        directory: str = "",
        pattern=None,
    ):
        self.on = on
        self.params = Parameters(inifile=inifile, code=code, directory=directory)
        self._read = self.params.loadSimuFile(
            self.on,
            geometry=geometry,
            cell="edges",
            pattern=pattern,
        )
        self.native_geometry = self._read.geometry
        self.dict = self._read.data
        self.coords = Coordinates(
            self.native_geometry, self._read.x1, self._read.x2, self._read.x3
        )
        for key in self.dict:
            self.dict[key] = GasField(
                key,
                self.dict[key],
                self.coords,
                self.native_geometry,
                self.on,
                "",
                inifile=inifile,
                code=code,
                directory=directory,
            )

    @classmethod
    def from_npy(
        cls,
        on: int,
        *,
        operation: str,
        directory=".",
        inifile: str = "",
        code: str = "",
    ):
        self = super().__new__(cls)
        self.on = on
        self.params = Parameters(inifile=inifile, code=code, directory=directory)
        self.dict = {}
        for dirname, dirs, files in os.walk(directory):
            if dirname == directory:
                fields = dirs
            else:
                for field in fields:
                    npyname = f"_{operation}_{field.upper()}.{self.on:04d}.npy"
                    if dirname == os.path.join(directory, field) and npyname in files:
                        headername = os.path.join(
                            directory, "header", f"header_{operation}.npy"
                        )
                        with open(headername, "rb") as file:
                            dict_coords = np.load(file, allow_pickle=True).item()

                        (
                            self.native_geometry,
                            coord0,
                            coord1,
                            coord2,
                        ) = dict_coords.values()
                        self.coords = Coordinates(
                            self.native_geometry, coord0, coord1, coord2
                        )

                        fileout = os.path.join(dirname, npyname)
                        with open(fileout, "rb") as file:
                            ret_data = np.load(file, allow_pickle=True)

                        self.dict[field.upper()] = GasField(
                            field.upper(),
                            ret_data,
                            self.coords,
                            self.native_geometry,
                            self.on,
                            operation=operation,
                            directory=directory,
                        )
        if not self.dict:
            raise FileNotFoundError(
                f"Original output was not reduced, or file '_{operation}_*.{self.on:04d}.npy'"
                " not recognized. Try with classical GasDataSet."
            )
        return self

    def __getitem__(self, key):
        if key in self.dict:
            return self.dict[key]
        else:
            raise KeyError

    def keys(self):
        """
        Returns
        =======
        keys of the dict
        """
        return self.dict.keys()

    def values(self):
        """
        Returns
        =======
        values of the dict
        """
        return self.dict.values()

    def items(self):
        """
        Returns
        =======
        items of the dict
        """
        return self.dict.items()
