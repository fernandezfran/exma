# exma

> **exma** is an extendable molecular analyzer.

[![Build Status](https://travis-ci.com/fernandezfran/exma.svg?branch=master)](https://travis-ci.com/github/fernandezfran/exma)
[![Documentation Status](https://readthedocs.org/projects/exma/badge/?version=latest)](https://exma.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/exma)](https://pypi.org/project/exma/)

This project is mainly developed with the objective of analyzing molecular dynamics trajectories using different tools that characterize both the configurational properties and the dynamics of the system. It is intended to analyze monatomic materials or alloys.


## Features

With _exma_ you can:

* Read and write MD trajectory frames in .xyz and .lammpstrj format (including charges, image of the particles or some other property).
* Write input file for LAMMPS.
* Estimate errors with the block average method.
* Initializate positions of atoms in SC, BCC or FCC crystals.
* Apply PBC.
* Calculate:
    - clusterization,
    - the coordination number (CN), the ligancy or in a shell, for monoatomic or diatomic systems,
    - effective neighbors (EN),
    - mean square displacement (MSD),
    - the radial distribution function (RDF) of monoatomic or diatomic systems,


## Requirements

_exma_ requires Python 3.8 to run, the others specific requirements are provided during installation. gcc 9.3 is necesary to compile the shared libreries if you are going to run `tox`.


## Installation

### Stable release

To install the most recent stable release of _exma_ with [pip](https://pip.pypa.io/en/stable/), run the following command in your termninal:

```bash
pip3 install exma
```

### From sources

To installing _exma_ from sources you can clone this [GitHub repo](https://github.com/fernandezfran/exma) 

```bash
git clone https://github.com/fernandezfran/exma.git
```

and inside your local directory install it in the following way 

```bash
pip3 install .
```

or

```bash
python3 setup.py install
```

## Usage

To use the extendable molecular analyzer in a project:

```python
import exma
```

There are different examples in the documentation [tutorials](https://exma.readthedocs.io/en/latest/tutorial.html).

## Release History

* 0.1.0 (2021-03-22)
    * First realease on PyPI
* 0.0.1 (2021-02)
    * Work in progress


## License

[MIT License](https://github.com/fernandezfran/exma/blob/master/LICENSE)


## TODO

- [ ] triclinic (non-orthogonal) boxes for the different modules.


## Contributing

1. Fork it: <https://github.com/fernandezfran/exma/fork>
2. Clone your fork locally: `git clone git@github.com:your_name/exma.git`
3. Install _exma_ and test it to check if everything works fine (you can run `tox` in the main directory after run `make` in each subdirectory of `exma/`).
4. Create a new branch with your feature: `git checkout -b your_feature`
5. Commit your changes: `git commit -am 'a detailed description of your feature'`
6. Push the branch: `git push origin your_feature`
7. Create a new Pull Request
