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

from ._trajectory_rw import TrajectoryReader

# ============================================================================
# CLASSES
# ============================================================================


class XYZ(TrajectoryReader):
    """Class to read xyz files.

    Parameters
    ----------
    filename : str
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

    def __init__(self, filename, ftype="xyz"):
        if ftype not in ["xyz", "property", "image"]:
            raise ValueError("ftype must be 'xyz', 'property' or 'image'")

        super().__init__(filename, ftype)

    def read_frame(self):
        """Read the actual frame of an .xyz file.

        Returns
        -------
        frame : `exma.core.AtomicSystem`
            This have all the information of the configurations of the system.

        Raises
        ------
        EOFError
            If there are no more frames to read
        """
        natoms = self.file_traj_.readline()
        if not natoms:
            raise EOFError("There is no more frames to read")

        natoms = np.intc(natoms)
        self.file_traj_.readline()  # usually a comment in .xyz files

        atom_type = []
        x, y, z = [], [], []
        q = [] if self.ftype == "property" else None
        ix, iy, iz = [], [], [] if self.ftype == "image" else None
        for i in range(natoms):
            xyzline = self.file_traj_.readline().split()

            atom_type.append(xyzline[0])

            x.append(xyzline[1])
            y.append(xyzline[2])
            z.append(xyzline[3])

            if self.ftype == "property":
                q.append(xyzline[4])
            elif self.ftype == "image":
                ix.append(xyzline[4])
                iy.append(xyzline[5])
                iz.append(xyzline[6])

        self.frame.natoms = natoms
        self.frame.types = np.asarray(atom_type, dtype=str)
        self.frame.x = np.asarray(x, dtype=np.float32)
        self.frame.y = np.asarray(y, dtype=np.float32)
        self.frame.z = np.asarray(z, dtype=np.float32)

        self.frame.ix = (
            np.asarray(ix, dtype=np.intc) if self.ftype == "image" else None
        )
        self.frame.iy = (
            np.asarray(iy, dtype=np.intc) if self.ftype == "image" else None
        )
        self.frame.iz = (
            np.asarray(iz, dtype=np.intc) if self.ftype == "image" else None
        )

        self.frame.q = (
            np.asarray(q, dtype=np.float32)
            if self.ftype == "property"
            else None
        )

        return self.frame


class LAMMPS(TrajectoryReader):
    """Class to read lammpstrj files.

    Parameters
    ----------
    filename : str
        name of the file where the trajectories of lammps are

    headerint : list, default=["idx", "types", "ix", "iy", "iz"]
        the columns that have int data types, the others are considered floats.
    """

    def __init__(self, filename, headerint=["idx", "types", "ix", "iy", "iz"]):
        super().__init__(filename, None)
        self.headerint = headerint

    def read_frame(self):
        """Read the actual frame of an .lammpstrj file.

        Returns
        -------
        frame : `exma.core.AtomicSystem`
            This have all the information of the configurations of the system.

        Raises
        ------
        EOFError
            If there are no more frames to read
        """
        comment = self.file_traj_.readline()
        if not comment:
            raise EOFError("There is no more frames to read")
        self.file_traj_.readline()
        self.file_traj_.readline()

        natoms = np.intc(self.file_traj_.readline())

        self.file_traj_.readline()

        box_size = []
        for _ in range(3):
            lohi = self.file_traj_.readline().split()
            box_size.append(np.float32(lohi[1]) - np.float32(lohi[0]))
        box = np.array(box_size)

        keys = self.file_traj_.readline().split()[2:]

        # make sure that the keywords id and type of lammps are not used in
        # self.frame.
        keys = list(
            map(
                lambda x: x.replace("id", "idx").replace("type", "types"), keys
            )
        )

        frame = {key: list() for key in keys}
        for _ in range(natoms):
            line = self.file_traj_.readline().split()
            for j, element in enumerate(line):
                frame[keys[j]].append(element)

        # a way to know the type of data (always a np.float32 except when the
        # data corresponds with integers: `id`, `type`, `ix`, `iy`, `iz`)
        dtypes = [
            np.float32 if key not in self.headerint else np.intc
            for key in keys
        ]

        self.frame.natoms = natoms
        self.frame.box = box
        for key, value, dtype in zip(frame.keys(), frame.values(), dtypes):
            self.frame.__dict__[key] = np.array(value, dtype=dtype)

        return self.frame


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
