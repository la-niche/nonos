# nonos
[![PyPI](https://img.shields.io/pypi/v/nonos.svg?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/nonos/)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/nonos)](https://pypi.org/project/nonos/)
[![Documentation Status](https://readthedocs.org/projects/nonos/badge/?version=latest)](https://nonos.readthedocs.io/en/latest/?badge=latest)

`nonos` is a Python 2D visualization library for planet-disk numerical simulations.

It supports vtk-formatted data from Pluto and Idefix, and dat-formatted data for Fargo-adsg and Fargo3D.

This page illustrates basic examples.
For more, read [the documentation !](https://nonos.readthedocs.io/en/latest/?badge=latest)

##### Data Formats
`nonos` supports the following data formats
- Pluto and Idefix: `data.*.vtk`
- Fargo-adsg: `gasdens.dat`, `gasvy*.dat`, `gasvx*.dat`
- Fargo3D: same as Fargo-adsg + `gasvz*.dat`, `gasenergy*.dat`

## Development status

`nonos` is considered public beta software: we are actively improving the
design and API, but we are not at the point where we want to bless the
current state as stable yet. We *are* trying to keep breaking changes to a
minimum, and run deprecation cycles to minimize the pain, however they might
happen in any minor release, so if you rely on `nonos` for your own work
(thank you !), we strongly encourage you to follow along releases and
upgrade frequently, so we have more opportunities to discuss if something
breaks.

## Installation

Get `nonos` and its minimal set of dependencies as

```shell
python -m pip install nonos
```

Optionally, you can install with the companion command line interface too
```shell
python -m pip install "nonos[cli]"
```

or, to get *all* optional dependencies (CLI included)
```shell
python -m pip install "nonos[all]"
```

## Examples

## Building a 2D map

We'll start by defining a `GasDataSet` object
```py
from nonos.api import GasDataSet

ds = GasDataSet(
    43,
    geometry="polar",
    directory="tests/data/idefix_planet3d",
)
```

We can select the `RHO` field, reduce it to a vertical
slice in the midplane, and derive a `Plotable` object mapping the cartesian `'x', 'y'` plane,
all while ensuring the 0th planet lies close to azimuth 0
```py
p = ds["RHO"].vertical_at_midplane().map("x", "y", rotate_with="planet0.dat")
```

Now let's actually visualize our results
```py
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.set_aspect("equal")

p.plot(
    fig,
    ax,
    log=True,
    cmap="inferno",
    title=r"$\rho_{\rm mid}$",
)
```

## Visualizing a 1D graph

This time, we'll reduce the `RHO` field to a single dimension,
via a latitudinal projection, followed by an azimuthal average,
and map the result to the radial axis.
```py
fig, ax = plt.subplots()
(
    ds["RHO"]
    .latitudinal_projection(theta=3 * 0.05)
    .azimuthal_average()
    .map("R")
    .plot(fig, ax, color="black", title=r"$\Sigma$")
)
```


### Reusing `nonos`' style

`nonos` comes with a matplotlib stylesheet, used by `nonos-cli`, that can be reused
programmatically, without importing the package, using matplotlib API
```python
import matplotlib.style

matplotlib.style.use("nonos.default")
```

See [`matplotlib.style`'s documentation](https://matplotlib.org/stable/api/style_api.html) for more.
