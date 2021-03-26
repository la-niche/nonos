# nonos
A tool to analyze results from idefix/pluto/fargo3d simulations (for protoplanetary disks more specifically, 2D ok, 3D cylindrical)
TODO: write readme for the command line tool mode (ex: nonos -diff -on 10) / local mode (nonos -l + config.toml file)

to be tested: the function/class part of the programm (cf the example here to start with)

error: ````streamlines```` & ````midplane=False```` -> not yet implemented

warning: ````isPlanet=False```` & ````corotate=True```` -> we don't rotate the grid if there is no planet for now.\nomegagrid = 0.

warning: ````cartesian=False```` & ````midplane=False```` -> plot not optimized for now in the (R,z) plane in polar.

#### Example of field
````python
from nonos import InitParamNonos, FieldNonos
init = InitParamNonos(info=True) #info=True gives the default parameters in the param file config.toml
fieldon = FieldNonos(init, field='RHO', on=25, diff=True)
````
#### Example of plot without streamlines
````python
from nonos import InitParamNonos, PlotNonos
import matplotlib.pyplot as plt

init = InitParamNonos()
ploton = PlotNonos(init, field='RHO', on=25, diff=True)

fig, ax = plt.subplots()
ploton.plot(ax, cartesian=True, fontsize=8)
plt.show()
````
#### Example of plot with streamlines with a planet
````python
from nonos import InitParamNonos, FieldNonos, PlotNonos, StreamNonos
import matplotlib.pyplot as plt

init = InitParamNonos(isPlanet=True, corotate=True)

ploton = PlotNonos(init, field='RHO', on=25, diff=True)
streamon = StreamNonos(init, on=25)
vx1on = FieldNonos(init, field='VX1', on=25, diff=False, log=False)
vr = vx1on.data
vx2on = FieldNonos(init, field='VX2', on=25, diff=False, log=False)
vphi = vx2on.data
vphi -= vx2on.omegaplanet[25]*vx2on.xmed[:,None,None]
streams = streamon.get_random_streams(vr,vphi,xmin=0.7,xmax=1.3, n=30)

fig, ax = plt.subplots()
ploton.plot(ax, cartesian=True, fontsize=8)
streamon.plot_streams(ax,streams,cartesian=True, color='k', linewidth=2, alpha=0.5)
plt.show()
````
#### Example of plot with a comparison between several simulations
````python
from nonos import InitParamNonos, PlotNonos
import matplotlib.pyplot as plt

dirpath=['path_to_dir1', 'path_to_dir2', 'path_to_dir3']

fig, ax = plt.subplots(figsize=(9,2.5), ncols=len(dirpath))
plt.subplots_adjust(left=0.05, right=0.94, top=0.95, bottom=0.1, wspace=0.4)
for i in range(len(dirpath)):
    init = InitParamNonos(directory=dirpath[i])
    ploton = PlotNonos(init, field='RHO', on=10, diff=True, directory=dirpath[i])
    ploton.plot(ax[i], cartesian=True, fontsize=6)

plt.show()
````
