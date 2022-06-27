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

from ._trajectory_rw import TrajectoryWriter

# ============================================================================
# CLASSES
# ============================================================================


class XYZ(TrajectoryWriter):
    """Class to write xyz files.

    Parameters
    ----------
    filename : str
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

    def __init__(self, filename, ftype="xyz"):
        if ftype not in ("xyz", "property", "image"):
            raise ValueError("ftype must be 'xyz', 'property' or 'image'")

        super(XYZ, self).__init__(filename, ftype)

    def write_frame(self, frame):
        """Write the actual frame in an .xyz file.

        Parameters
        ----------
        frame : `exma.core.AtomicSystem`
            This have all the information of the configurations of the system.
        """
        self.file_traj_.write(f"{frame.natoms}\n\n")

        for i in range(frame.natoms):
            line = f"{frame.types[i]:s}  "
            line += f"{frame.x[i]:.6e}  "
            line += f"{frame.y[i]:.6e}  "
            line += f"{frame.z[i]:.6e}"

            if self.ftype == "property":
                line += f"  {frame.q[i]:.6e}"
            if self.ftype == "image":
                line += f"  {frame.ix[i]:d}"
                line += f"  {frame.iy[i]:d}"
                line += f"  {frame.iz[i]:d}"

            line += "\n"
            self.file_traj_.write(line)


class LAMMPS(TrajectoryWriter):
    """Class to write lammpstrj files.

    Parameters
    ----------
    filename : str
        name of the file where the trajectories of lammps are going to be
        written
    """

    def __init__(self, filename, ftype="custom"):
        super(LAMMPS, self).__init__(filename, ftype)
        self.timestep = 0

    def write_frame(self, frame):
        """Write the actual frame in a .lammpstrj file.

        Parameters
        ----------
        frame : `exma.core.AtomicSystem`
            This have all the information of the configurations of the system.
        """
        self.file_traj_.write("ITEM: TIMESTEP\n")
        self.file_traj_.write(f"{self.timestep:d}\n")
        self.file_traj_.write("ITEM: NUMBER OF ATOMS\n")
        self.file_traj_.write(f"{frame.natoms:d}\n")
        self.file_traj_.write("ITEM: BOX BOUNDS pp pp pp\n")
        for i in range(0, 3):
            self.file_traj_.write(f"0.0\t{frame.box[i]:.6e}\n")

        self.timestep += 1

        traj_header = "ITEM: ATOMS"
        for key in frame.__dict__.keys():
            if key.startswith("_") or key in ("natoms", "box"):
                continue
            elif frame.__dict__[key] is not None:
                traj_header += " " + str(key).replace("idx", "id").replace(
                    "types", "type"
                )
        traj_header += "\n"
        self.file_traj_.write(traj_header)

        for i in range(frame.natoms):
            line = ""
            for key in frame.__dict__.keys():
                if key.startswith("_") or key in ("natoms", "box"):
                    continue
                if frame.__dict__[key] is not None:
                    value = frame.__dict__[key][i]
                    line += (
                        f"{value:.6e}  "
                        if isinstance(value, np.float32)
                        else f"{value}  "
                    )
            line = line.rstrip()
            line += "\n"
            self.file_traj_.write(line)


# ============================================================================
# FUNCTIONS
# ============================================================================


def in_lammps(file_in, frame):
    """Write a frame as an input file for LAMMPS.

    Parameters
    ----------
    file_in : str
        name of the file where you want to write the input info

    frame : `exma.core.AtomicSystem`
        This have all the information of the configurations of the system.
    """
    header = ("natoms", "box")
    with open(file_in, "w") as f_in:

        f_in.write("# the first three lines are comments...\n")

        f_in_header = "# columns in order:"
        for key in frame.__dict__.keys():
            if key.startswith("_") or key in ("natoms", "box"):
                continue
            elif frame.__dict__[key] is not None:
                f_in_header += " " + str(key).replace("idx", "id").replace(
                    "types", "type"
                )
        f_in_header += "\n"
        f_in.write(f_in_header)

        f_in.write("# input file for LAMMPS generated by exma\n")

        f_in.write(f"{frame.natoms:d} atoms\n")

        noat = np.unique(frame.types).size  # number of atom types
        f_in.write(f"{noat:d} atom types\n\n")

        f_in.write(f"0.0 \t {frame.box[0]:.6e} \t xlo xhi\n")
        f_in.write(f"0.0 \t {frame.box[1]:.6e} \t ylo yhi\n")
        f_in.write(f"0.0 \t {frame.box[2]:.6e} \t zlo zhi\n\n")

        f_in.write("Atoms\n\n")

        for i in range(frame.natoms):
            line = ""
            for key in frame.__dict__.keys():
                if key.startswith("_") or key in header:
                    continue
                if frame.__dict__[key] is not None:
                    value = frame.__dict__[key][i]
                    line += (
                        f"{value:.6e}  "
                        if isinstance(value, np.float32)
                        else f"{value}  "
                    )
            line = line.rstrip()
            line += "\n"
            f_in.write(line)
