# exma

[![Github Actions CI](https://github.com/fernandezfran/exma/actions/workflows/exma_ci.yml/badge.svg)](https://github.com/fernandezfran/exma/actions/workflows/exma_ci.yml)
[![Documentation Status](https://readthedocs.org/projects/exma/badge/?version=latest)](https://exma.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/exma)](https://pypi.org/project/exma/)

**exma** is a Python library with C extensions to analyze and manipulate 
molecular dynamics trajectories and electrochemical data.


## Main Features

Some of the main calculations that can be performed are listed below:

* mean square displacement,
* radial distribution function,
* coordination number, the ligancy or in a shell,

among others.

For more precise information, see the [tutorials](https://exma.readthedocs.io/en/latest/tutorial.html)
and the [API](https://exma.readthedocs.io/en/latest/api.html) in the documentation.


## Requirements

You need Python 3.8+ to run exma. 


## Installation

### Stable release

To install the most recent stable release of exma with [pip](https://pip.pypa.io/en/stable/), 
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
