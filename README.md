# exma

> **exma** is an extendable molecular analyzer.

[![Build Status](https://travis-ci.com/fernandezfran/exma.svg?branch=master)](https://travis-ci.com/github/fernandezfran/exma)
[![Documentation Status](https://readthedocs.org/projects/exma/badge/?version=latest)](https://exma.readthedocs.io/en/latest/?badge=latest)

This project is mainly developed with the objective of analyzing molecular dynamics trajectories using different tools that characterize both the configurational properties and the dynamics of the system. It is intended to analyze monatomic materials or alloys.


## Features

With _exma_ you can:

* Read and write MD trajectory frames in .xyz and .lammpstrj format (including charges, image of the particles or some other property).
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

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# in this script the coordination numbers of sc, bcc and fcc crystals are 
#   calculated, the steps are following for each system:
#
#       1. with exma.atoms.positions generate the positions of the atoms
#           the first argument is the number of atoms and the second the box size
#       2. with exma.cn.monoatomic initializate the CN calculation
#           the first argument is the number of atoms, the second the box size
#           in each direction and the third is a cutoff distance
#       3. with CN.accumulate calculate de CN of this "frame"
#           the argument is the positions of the atoms
#       4. with CN.end return the CN of each atom
#           the first argument is the atom type to be written (not interested here)
#           the second are the positions and the third is False to not write an
#           output file
#
import numpy as np
import exma

sc = exma.atoms.positions(8, 4.0).sc()
cn_sc = exma.cn.coordination_number.monoatomic(8, 3.0)
cn_sc.accumulate(np.full(3,4.0), sc)
cn_sc = np.mean(cn_sc.end(0, sc, False))

bcc = exma.atoms.positions(16, 4.0).bcc()
cn_bcc = exma.cn.coordination_number.monoatomic(16, 1.8)
cn_bcc.accumulate(np.full(3,4.0), bcc)
cn_bcc = np.mean(cn_bcc.end(0, bcc, False))

fcc = exma.atoms.positions(32, 4.0).fcc()
cn_fcc = exma.cn.coordination_number.monoatomic(32, 1.5)
cn_fcc.accumulate(np.full(3,4.0), fcc)
cn_fcc = np.mean(cn_fcc.end(0, fcc, False))

print("#  %d is the coordination number of a sc crystal" % cn_sc)
print("#  %d is the coordination number of a bcc crystal" % cn_bcc)
print("# %d is the coordination number of a fcc crystal" % cn_fcc)
```

There are more examples in the Documentation.


## Release History

* 0.1.0
    * First realease on PyPI
* 0.0.1
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
