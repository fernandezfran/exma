# exma

**exma** is an extendable molecular analyzer.

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


## License

[MIT License](https://choosealicense.com/licenses/mit/)
