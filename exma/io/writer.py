#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Classes and functions to write MD trayectories."""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np

from ..core import TrajectoryWriter

# ============================================================================
# CLASSES
# ============================================================================


class XYZ(TrajectoryWriter):
    """Class to write xyz files.

    Parameters
    ----------
    file_traj : str
        name of the file where the trajectories in xyz format are going to
        be written

    ftype : str, default="xyz"
        the possible values are `xyz`, `property` and `image`. `xyz`
        if is the usual xyz file. `property` if in the last column there is
        a property. `image` if in the last three columns there are the image
        box of the corresponding atom.

    Raises
    ------
    ValueError
        If xyz file type is not among the possible values
    """

    def __init__(self, file_traj, ftype="xyz"):
        if ftype not in ["xyz", "property", "image"]:
            raise ValueError("ftype must be 'xyz', 'property' or 'image'")

        super(XYZ, self).__init__(file_traj, ftype)

    def write_frame(self, frame):
        """Write the actual frame in an .xyz file.

        Parameters
        ----------
        frame : dict
            with the keys `natoms`, `type`, `x`, `y`, `z`, the number of
            atoms in the frame (int), the element of each atom (str) and
            the x, y, z positions of the atoms (np.array), respectively. If
            `ftype="property"` was selected, then the key `property` is
            also a key. On the other hand, if `ftype="image"` was selected,
            then `ix`, `iy` and `iz` are keys and have the positions images
            in each direction (np.array), respectively.
        """
        natoms = frame["natoms"]
        self.file_traj.write(f"{natoms}\n\n")

        for i in range(natoms):
            line = f"{frame['type'][i]:s}  "
            line += f"{frame['x'][i]:.6e}  "
            line += f"{frame['y'][i]:.6e}  "
            line += f"{frame['z'][i]:.6e}"

            if self.ftype == "property":
                line += f"  {frame['property'][i]:.6e}"
            if self.ftype == "image":
                line += f"  {frame['ix'][i]:d}"
                line += f"  {frame['iy'][i]:d}"
                line += f"  {frame['iz'][i]:d}"

            line += "\n"
            self.file_traj.write(line)


class LAMMPS(TrajectoryWriter):
    """Class to write lammpstrj files.

    Parameters
    ----------
    file_traj : str
        name of the file where the trajectories of lammps are going to be
        written
    """

    def __init__(self, file_traj, ftype="custom"):
        super(LAMMPS, self).__init__(file_traj, ftype)
        self.timestep = 0

    def write_frame(self, frame):
        """Write the actual frame in a .lammpstrj file.

        Parameters
        ----------
        dict
            with the list of attributes selected by the `dump` command of
            LAMMPS for each atom as keys and the corresponding frame values
            as `np.array`; except the number of atoms, which is an `int`.
            `natoms` and `box` must be defined in the dict.
        """
        self.file_traj.write("ITEM: TIMESTEP\n")
        self.file_traj.write(f"{self.timestep:d}\n")
        self.file_traj.write("ITEM: NUMBER OF ATOMS\n")
        self.file_traj.write(f"{frame['natoms']:d}\n")
        self.file_traj.write("ITEM: BOX BOUNDS pp pp pp\n")
        for i in range(0, 3):
            self.file_traj.write(f"0.0\t{frame['box'][i]:.6e}\n")

        self.timestep += 1

        traj_header = "ITEM: ATOMS"
        for key in frame.keys():
            if key in ["natoms", "box"]:
                continue
            traj_header += " " + str(key)
        traj_header += "\n"
        self.file_traj.write(traj_header)

        for i in range(frame["natoms"]):
            line = ""
            for key in frame.keys():
                if key in ["natoms", "box"]:
                    continue
                value = frame[key][i]
                line += (
                    f"{value:.6e}  "
                    if isinstance(value, np.float32)
                    else f"{value}  "
                )
            line = line.rstrip()
            line += "\n"
            self.file_traj.write(line)


# ============================================================================
# FUNCTIONS
# ============================================================================


def in_lammps(file_in, frame):
    """Write a frame as an input file for LAMMPS.

    Parameters
    ----------
    file_in : str
        name of the file where you want to write the input info

    frame : dict
        with the list of attributes that you want to read with `read_data`
        in LAMMPS. `natoms` and `box` must be defined in the dict.
    """
    header = ["natoms", "box"]
    with open(file_in, "w") as f_in:

        keys = [key for key in frame.keys() if key not in header]

        f_in.write("# the first three lines are comments...\n")
        f_in.write(f"# columns in order: {' '.join(keys)}\n")
        f_in.write("# input file for LAMMPS generated by exma\n")

        f_in.write(f"{frame['natoms']:d} atoms\n")

        noat = np.unique(frame["type"]).size  # number of atom types
        f_in.write(f"{noat:d} atom types\n\n")

        f_in.write(f"0.0 \t {frame['box'][0]:.6e} \t xlo xhi\n")
        f_in.write(f"0.0 \t {frame['box'][1]:.6e} \t ylo yhi\n")
        f_in.write(f"0.0 \t {frame['box'][2]:.6e} \t zlo zhi\n\n")

        f_in.write("Atoms\n\n")

        for i in range(frame["natoms"]):
            line = ""
            for key in frame.keys():
                if key in header:
                    continue
                value = frame[key][i]
                line += (
                    f"{value:.6e}  "
                    if isinstance(value, np.float32)
                    else f"{value}  "
                )
            line = line.rstrip()
            line += "\n"
            f_in.write(line)
