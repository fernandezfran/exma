#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ======================================================================
# DOCS
# ======================================================================

"""exma IO module for interacting with molecular dynamics files."""

# ======================================================================
# IMPORTS
# ======================================================================

import numpy as np

from . import reader
from . import writer

# ======================================================================
# FUNCTIONS
# ======================================================================


def xyz2lammpstrj(xyztraj, lammpstrj_name, cell_info, xyzftype="xyz"):
    """Rewrite an xyz file to a lammpstrj file.

    Parameters
    ----------
    xyztraj : str
        the name of the file with the xyz trajectory.

    lammpstrj_name : str
        the name of the file with the lammpstrj trajectory.

    cell_info : dict
        with the `box`, the lenght of the box in each direction, another
        dictionary identified with the `type` key that has within it a
        correspondence between the elements present in xyz file with integer
        identification numbers, e.g. {"Sn": 1, "O": 2}

    xyzftype : str
        the `ftype` of xyz file.
    """
    xyz = reader.XYZ(xyztraj, xyzftype)
    lmp = writer.LAMMPS(lammpstrj_name)
    try:
        while True:
            xyz_frame = xyz.read_frame()

            xyz_frame["type"] = [
                cell_info["type"][t] for t in xyz_frame["type"]
            ]
            xyz_frame = {
                key: value
                for key, value in zip(xyz_frame.keys(), xyz_frame.values())
                if value is not None
            }
            cell_info["id"] = np.arange(1, xyz_frame["natoms"] + 1)
            del cell_info["type"]

            lmp.write_frame(dict(cell_info, **xyz_frame))

    except EOFError:
        xyz.file_close()
        lmp.file_close()


def xyz2inlmp():
    """Not implemented yet."""
    raise NotImplementedError("To be implemented soon.")


def lammpstrj2xyz(lammpstrjtraj, xyz_name, type_info):
    """Rewrite a lammpstrj file to an xyz file.

    Parameters
    ----------
    lammpstrjtraj : str
        the name of the file with the lammpstrj trajectory.

    xyz_name : str
        the name of the file with the lammpstrj trajectory.

    type_info : dict
        a correspondence between the elements id present in lammpstrj file
        with str element, e.g. {1: "Sn", 2: "O"}
    """
    lmp = reader.LAMMPS(lammpstrjtraj)
    xyz = writer.XYZ(xyz_name)
    try:
        while True:
            lmp_frame = lmp.read_frame()
            lmp_frame["type"] = [type_info[t] for t in lmp_frame["type"]]

            xyz_frame = {
                key: value
                for key, value in zip(lmp_frame.keys(), lmp_frame.values())
                if key in ["natoms", "type", "x", "y", "z", "ix", "iy", "iz"]
            }
            xyz.write_frame(xyz_frame)

    except EOFError:
        lmp.file_close()
        xyz.file_close()


def lammpstrj2inlmp():
    """Not implemented yet."""
    raise NotImplementedError("To be implemented soon.")


def cif2xyz():
    """Not implemented yet."""
    raise NotImplementedError("To be implemented soon.")


def cif2inlmp():
    """Not implemented yet."""
    raise NotImplementedError("To be implemented soon.")
