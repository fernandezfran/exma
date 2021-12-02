#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Classes and functions to read MD trayectories."""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np

import pandas as pd

from ..core import TrajectoryReader

# ============================================================================
# CLASSES
# ============================================================================


class XYZ(TrajectoryReader):
    """Class to read xyz files.

    Parameters
    ----------
    file_traj : str
        name of the file where the trajectories in xyz format are

    ftype : str, default="xyz"
        the possible values are `xyz`, `property` and `image`.
        `xyz` if is the usual xyz file. `property` if in the last
        column there is a property. `image` if in the last three columns
        there are the image box of the corresponding atom.

    Raises
    ------
    ValueError
        If xyz file type is not among the possible values
    """

    def __init__(self, file_traj, ftype="xyz"):
        if ftype not in ["xyz", "property", "image"]:
            raise ValueError("ftype must be 'xyz', 'property' or 'image'")

        super(XYZ, self).__init__(file_traj, ftype)

    def read_frame(self):
        """Read the actual frame of an .xyz file.

        Returns
        -------
        dict
            frame with the keys `natoms`, `type`, `x`, `y`, `z`, the number
            of atoms (int), the element of each atom (np array of strings)
            and the x, y, z positions of the atoms (np.array), respectively.
            If `ftype="property"` was selected, then the key `property` is
            also a key. On the other hand, if `ftype="image"` was selected,
            then `ix`, `iy` and `iz` are keys and have the positions images
            in each direction (np.array), respectively.

        Raises
        ------
        EOFError
            If there are no more frames to read
        """
        natoms = self.file_traj.readline()
        if not natoms:
            raise EOFError("There is no more frames to read")

        natoms = np.intc(natoms)
        self.file_traj.readline()  # usually a comment in .xyz files

        atom_type = []
        x, y, z = [], [], []
        prop = [] if self.ftype == "property" else None
        ix, iy, iz = [], [], [] if self.ftype == "image" else None
        for i in range(natoms):
            xyzline = self.file_traj.readline().split()

            atom_type.append(xyzline[0])

            x.append(xyzline[1])
            y.append(xyzline[2])
            z.append(xyzline[3])

            if self.ftype == "property":
                prop.append(xyzline[4])
            elif self.ftype == "image":
                ix.append(xyzline[4])
                iy.append(xyzline[5])
                iz.append(xyzline[6])

        frame = {
            "natoms": natoms,
            "type": np.asarray(atom_type, dtype=str),
            "x": np.asarray(x, dtype=np.float32),
            "y": np.asarray(y, dtype=np.float32),
            "z": np.asarray(z, dtype=np.float32),
        }
        frame["property"] = (
            np.asarray(prop, dtype=np.float32)
            if self.ftype == "property"
            else None
        )
        frame["ix"] = (
            np.asarray(ix, dtype=np.intc) if self.ftype == "image" else None
        )
        frame["iy"] = (
            np.asarray(iy, dtype=np.intc) if self.ftype == "image" else None
        )
        frame["iz"] = (
            np.asarray(iz, dtype=np.intc) if self.ftype == "image" else None
        )

        return frame


class LAMMPS(TrajectoryReader):
    """Class to read lammpstrj files.

    Parameters
    ----------
    file_traj : str
        name of the file where the trajectories of lammps are

    headerint : list, default=["id", "type", "ix", "iy", "iz"]
        the columns that have int data types, the others are considered floats.
    """

    def __init__(self, file_traj, headerint=["id", "type", "ix", "iy", "iz"]):
        super(LAMMPS, self).__init__(file_traj, None)
        self.headerint = headerint

    def read_frame(self):
        """Read the actual frame of an .lammpstrj file.

        Returns
        -------
        dict
            frame with the list of attributes selected by the `dump` command
            of LAMMPS for each atom as keys and the corresponding frame values
            as `np.array`; except the number of atoms, which is an `int`.

        Raises
        ------
        EOFError
            If there are no more frames to read
        """
        comment = self.file_traj.readline()
        if not comment:
            raise EOFError("There is no more frames to read")
        self.file_traj.readline()
        self.file_traj.readline()

        natoms = np.intc(self.file_traj.readline())

        self.file_traj.readline()

        box_size = []
        for _ in range(3):
            lohi = self.file_traj.readline().split()
            box_size.append(np.float32(lohi[1]) - np.float32(lohi[0]))
        box = np.array(box_size)

        cell = {"natoms": natoms, "box": box}

        keys = self.file_traj.readline().split()[2:]
        frame = {key: list() for key in keys}
        for _ in range(natoms):
            line = self.file_traj.readline().split()
            for j, element in enumerate(line):
                frame[keys[j]].append(element)

        # a way to know the type of data (always a np.float32 except when the
        # data corresponds with integers: `id`, `type`, `ix`, `ìy`, `iz`)
        dtypes = [
            np.float32 if key not in self.headerint else np.intc
            for key in keys
        ]

        frame = {
            key: np.array(value, dtype=dtype)
            for key, value, dtype in zip(frame.keys(), frame.values(), dtypes)
        }

        return dict(cell, **frame)


# ============================================================================
# FUNCTIONS
# ============================================================================


def read_log_lammps(logname="log.lammps"):
    """Read log file of lammps.

    Parameters
    ----------
    logname : str, defalut="log.lammps".
        the name of the file where the thermodynamic info was logged.

    Returns
    -------
    pd.DataFrame
        A `pd.DataFrame` with the columns corresponding to the thermodynamic
        info.

    Notes
    -----
    It only works if the first thermo parameter is `Step`.

    """
    with open(logname, "r") as flog:
        # ignore all previous info
        line = flog.readline()
        while line.startswith("Step ") is False:
            line = flog.readline()

        keys = list(line.split())
        thermo = {key: list() for key in keys}

        # append info until subsequent info to be ignored
        line = flog.readline()
        while line.startswith("Loop time") is False:
            for i, element in enumerate(line.split()):
                thermo[keys[i]].append(element)
            line = flog.readline()

        thermo = {
            key: np.array(value, dtype=np.float32)
            for key, value in zip(thermo.keys(), thermo.values())
        }

        return pd.DataFrame(thermo)
