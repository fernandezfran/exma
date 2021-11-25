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


def xyz2lammsptrj(xyztraj, lammpstrj_name, cell_info, xyzftype="xyz"):
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

            cell_info["id"] = np.arange(1, xyz_frame["natoms"] + 1)
            xyz_frame["type"] = [
                cell_info["type"][t] for t in xyz_frame["type"]
            ]
            xyz_frame = {
                key: value
                for key, value in zip(xyz_frame.keys(), xyz_frame.values())
                if value is not None
            }

            lmp.write_frame(dict(cell_info, **xyz_frame))

    except EOFError:
        xyz.file_close()
        lmp.file_close()


def xyz2inlmp():
    """Not implemented yet."""
    raise NotImplementedError("To be implemented soon.")


def lammsptrj2xyz():
    """Not implemented yet."""
    raise NotImplementedError("To be implemented soon.")


def lammpstrj2inlmp():
    """Not implemented yet."""
    raise NotImplementedError("To be implemented soon.")


def cif2xyz():
    """Not implemented yet."""
    raise NotImplementedError("To be implemented soon.")


def cif2inlmp():
    """Not implemented yet."""
    raise NotImplementedError("To be implemented soon.")
