# Welcome to nonos

nonos is a 2D visualization Python library, for planet-disk numerical simulations, as well as a command line application. It works with VTK-formatted data from Pluto and Idefix, and dat-formatted data for Fargo-adsg and Fargo3D. This project and the documentation are currently under development.

## Data Formats

We list here the accepted formats for the data:

* Pluto and Idefix: `data.*.vtk`
* Fargo-adsg: `gasdens*.dat`, `gasvy*.dat`, `gasvx*.dat`
* Fargo3D: same as Fargo-adsg + `gasvz*.dat`, `gasenergy*.dat`

!!! info "Parameter file"

    In addition to the output files, nonos requires access to the parameter file in the same working directory. By default, these are idefix.ini for idefix, variables.par for fargo3d and pluto.ini for pluto.

## Installation

The easiest installation method is

```bash
$ python -m pip install nonos
```
