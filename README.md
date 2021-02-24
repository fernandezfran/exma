# exma

> **exma** is an extendable molecular analyzer.

## Features

With _exma_ you can:

* Read and write MD trajectory frames in .xyz and .lammpstrj format (including charges, image of the particles or some other property).
* Estimate errors with the block average method.
* Initializate positions of atoms in SC, BCC or FCC crystals.
* Apply PBC.
* Calculate:
    - the coordination number (CN), the ligancy or in a shell, for monoatomic or diatomic systems.
    - effective neighbors (EN).
    - mean square displacement (MSD).
    - the radial distribution function (RDF) of monoatomic or diatomic systems.
    - short range order (SRO): Warren-Cowley parameter (WCP).


## Requirements

_exma_ requires Python 3.8 to run, the others specific requirements are provided during installation. 


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
import exma
```

## Release History

* 0.1.0
    * First realease on PyPI
* 0.0.1
    * Work in progress


## License

[MIT License](https://choosealicense.com/licenses/mit/)


## TODO

- [ ] write rdf.accumulte in C and use ctypes.
- [ ] write cn.accumulte in C and use ctypes.
- [ ] write sro.warren_cowley.accumulte in C and use ctypes.


## Contibuting

1. Fork it: <https://github.com/fernandezfran/exma/fork>
2. Clone your fork locally: `git clone git@github.com:your_name/exma.git`
3. Install _exma_ and test it to check if everything works fine (you can run `tox` in the directory).
4. Create a new branch with your feature: `git checkout -b your_feature`
5. Commit your changes: `git commit -am 'a detailed description of your feature'`
6. Push the branch: `git push origin your_feature`
7. Create a new Pull Request
