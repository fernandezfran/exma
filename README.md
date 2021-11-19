# exma

[![Build Status](https://travis-ci.com/fernandezfran/exma.svg?branch=master)](https://travis-ci.com/github/fernandezfran/exma)
[![Documentation Status](https://readthedocs.org/projects/exma/badge/?version=latest)](https://exma.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/exma)](https://pypi.org/project/exma/)

**exma** is a library that allows to analyze molecular dynamics trajectories with
different tools.


## Main Features

Some of the main functionalities are listed below:

* Read and write MD files in xyz or lammpstrj format.
* Initializate positions of atoms in SC, BCC, FCF, DC crystals or spherical
nanoparticles.
* Calculate: 
    - mean square displacement,
    - radial distribution function,
    - coordination number, the ligancy or in a shell,
    - clusterization,
    - effective neighbors,
    - short range order, an amorphous parameter, defined [here](https://doi.org/10.1039/D1CP02216D),

For more precise information, read the [tutorials](https://exma.readthedocs.io/en/latest/tutorial.html)
and the [API](https://exma.readthedocs.io/en/latest/api.html) in the documentation.


## Requirements

You need Python 3.8+ to run **exma**. 


## Installation

### Stable release

To install the most recent stable release of **exma** with [pip](https://pip.pypa.io/en/stable/), 
run the following command in your termninal:

```bash
pip install exma
```

### From sources

To installing it from sources you can clone this [GitHub repo](https://github.com/fernandezfran/exma) 

```bash
git clone https://github.com/fernandezfran/exma.git
```

and inside your local directory install it in the following way 

```bash
pip install -e .
```


## License

[MIT License](https://github.com/fernandezfran/exma/blob/master/LICENSE)
