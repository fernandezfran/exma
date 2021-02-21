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

To installing _exma_ you can clone the GitHub repo [exma](https://github.com/fernandezfran/exma) and inside your local directory install it with [pip](https://pip.pypa.io/en/stable/)

```bash
pip3 install .
```

or you can use the package manager and install it by the usual way

```bash
pip3 install exma
```


## Usage

```python
import exma
```

or

```python
from exma import ... # replace the three dots by the module of interest
```
you can read the [Documentation](link.agregar) to see some examples.


## License

[MIT License](https://choosealicense.com/licenses/mit/)
