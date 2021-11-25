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

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

# ============================================================================
# CLASSES
# ============================================================================


class TrajectoryReader:
    """Class to read trajectory files.

    Parameters
    ----------
    file_traj : str
        name of the file where the trajectories in xyz format are

    ftype : str
        different type of files depending on the child class.
    """

    def __init__(self, file_traj, ftype):
        self.file_traj = open(file_traj, "r")
        self.ftype = ftype

    def read_frame(self):
        """Read the actual frame of the file."""
        raise NotImplementedError("Implemented in child classes.")

    def file_close(self):
        """Close the trayectory file."""
        self.file_traj.close()


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
            with the keys `natoms`, `type`, `x`, `y`, `z`, the number of
            atoms (int), the element of each atom (str) and the x, y, z
            positions of the atoms (np.array), respectively. If
            `ftype="property"` was selected, then the key `property` is
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

        dict_ = {
            "natoms": natoms,
            "type": atom_type,
            "x": np.asarray(x, dtype=np.float32),
            "y": np.asarray(y, dtype=np.float32),
            "z": np.asarray(z, dtype=np.float32),
        }
        dict_["property"] = (
            np.asarray(prop, dtype=np.float32)
            if self.ftype == "property"
            else None
        )
        dict_["ix"] = (
            np.asarray(ix, dtype=np.intc) if self.ftype == "image" else None
        )
        dict_["iy"] = (
            np.asarray(iy, dtype=np.intc) if self.ftype == "image" else None
        )
        dict_["iz"] = (
            np.asarray(iz, dtype=np.intc) if self.ftype == "image" else None
        )

        return dict_


class LAMMPS(TrajectoryReader):
    """Class to read lammpstrj files.

    Parameters
    ----------
    file_traj : str
        name of the file where the trajectories of lammps are
    """

    def __init__(self, file_traj, ftype="custom"):
        super(LAMMPS, self).__init__(file_traj, None)

    def read_frame(self):
        """Read the actual frame of an .lammpstrj file.

        Returns
        -------
        dict
            with the list of attributes selected by the `dump` command of
            LAMMPS for each atom as keys and the corresponding frame values
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

        cell_dict = {"natoms": natoms, "box": box}

        keys = self.file_traj.readline().split()[2:]
        frame_dict = {key: list() for key in keys}
        for _ in range(natoms):
            line = self.file_traj.readline().split()
            for j, element in enumerate(line):
                frame_dict[keys[j]].append(element)

        # automatic way to know the type of data (between int or float)
        dtypes = [
            np.float32 if "." in element else np.intc for element in line
        ]

        frame_dict = {
            key: np.array(value, dtype=dtype)
            for key, value, dtype in zip(
                frame_dict.keys(), frame_dict.values(), dtypes
            )
        }

        return dict(cell_dict, **frame_dict)
