#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""exma IO module for interacting with molecular dynamics files."""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np

from . import reader
from . import writer

# ============================================================================
# FUNCTIONS
# ============================================================================


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

    xyzftype : str, default="xyz"
        the `ftype` of xyz file.
    """
    frames = reader.read_xyz(xyztraj, xyzftype)

    for frame in frames:
        frame.box = cell_info["box"]
        frame.idx = np.arange(1, frame.natoms + 1)
        frame.types = [cell_info["type"][t] for t in frame.types]

    writer.write_lammpstrj(frames, lammpstrj_name)


def xyz2inlmp(xyztraj, inlammps_name, cell_info, nframe=-1, xyzftype="xyz"):
    """Write an xyz frame to an input data file of LAMMPS.

    Parameters
    ----------
    xyztraj : str
        the name of the file with the xyz trajectory.

    inlammps_name : str
        the name of the file to write to.

    cell_info : dict
        with the `box`, the lenght of the box in each direction, another
        dictionary identified with the `type` key that has within it a
        correspondence between the elements present in xyz file with integer
        identification numbers, e.g. {"Sn": 1, "O": 2}

    nframe : int, default=-1
        number of the frame to write, by default is -1, that is, the last.

    xyzftype : str, default="xyz"
        the `ftype` of xyz file.

    Raises
    ------
    IndexError
        If the number of the frame to write is not in the trajectory file.
    """
    frames = reader.read_xyz(xyztraj, xyzftype)
    try:
        frame = frames[nframe]
    except IndexError as exc:
        errmsg = f"frame {nframe} does not exist in the trajectory file."
        raise IndexError(errmsg) from exc

    frame.box = cell_info["box"]
    frame.idx = np.arange(1, frame.natoms + 1)
    frame.types = [cell_info["type"][t] for t in frame.types]
    frame.q = cell_info["q"] if "q" in cell_info.keys() else None

    writer.write_in_lammps(frame, inlammps_name)


def lammpstrj2xyz(lammpstrjtraj, xyz_name, type_info, xyzftype="xyz"):
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

    xyzftype : str, default="xyz"
        the `ftype` of xyz file.
    """
    frames = reader.read_lammpstrj(lammpstrjtraj)

    for frame in frames:
        frame = frame._sort() if not frame._sorted() else frame
        frame.types = [type_info[t] for t in frame.types]

    writer.write_xyz(frames, xyz_name, xyzftype)


def lammpstrj2inlmp(lammpstrjtraj, inlammps_name, nframe=-1):
    """Write a lammpstrj frame to an input data file of LAMMPS.

    Parameters
    ----------
    lammpstrjtraj : str
        the name of the file with the lammpstrj trajectory.

    inlammps_name : str
        the name of the file to write to.

    nframe : int, default=-1
        number of the frame to write, by default is -1, that is, the last.

    Raises
    ------
    IndexError
        If the number of the frame to write is not in the trajectory file.
    """
    frames = reader.read_lammpstrj(lammpstrjtraj)
    try:
        frame = frames[nframe]
    except IndexError as exc:
        errmsg = f"frame {nframe} does not exist in the trajectory file."
        raise IndexError(errmsg) from exc

    frame = frame._sort() if not frame._sorted() else frame

    writer.write_in_lammps(frame, inlammps_name)
