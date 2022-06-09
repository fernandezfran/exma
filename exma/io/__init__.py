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

import warnings

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
    with reader.XYZ(xyztraj, xyzftype) as xyz, writer.LAMMPS(
        lammpstrj_name
    ) as lmp:
        try:
            while True:
                xyz_frame = xyz.read_frame()

                xyz_frame.box = cell_info["box"]
                xyz_frame.idx = np.arange(1, xyz_frame.natoms + 1)
                xyz_frame.types = [
                    cell_info["type"][t] for t in xyz_frame.types
                ]

                lmp.write_frame(xyz_frame)

        except EOFError:
            ...


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
    """
    nframe = np.inf if nframe == -1 else nframe
    with reader.XYZ(xyztraj, xyzftype) as xyz:
        try:
            iframe = 0
            dframe = xyz.read_frame()
            while iframe < nframe:
                xyz_frame = xyz.read_frame()

                iframe += 1
                dframe = xyz_frame

        except EOFError:
            if nframe != np.inf:
                warnings.warn(
                    f"frame {nframe} does not exist in the trajectory file, "
                    f"therefore the last frame ({iframe}) was written."
                )

    dframe.box = cell_info["box"]
    dframe.idx = np.arange(1, dframe.natoms + 1)
    dframe.types = [cell_info["type"][t] for t in dframe.types]
    if "q" in cell_info.keys():
        dframe.q = cell_info["q"]

    writer.in_lammps(inlammps_name, dframe)


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
    with reader.LAMMPS(lammpstrjtraj) as lmp, writer.XYZ(
        xyz_name, xyzftype
    ) as xyz:
        try:
            while True:
                lmp_frame = lmp.read_frame()
                lmp_frame = (
                    lmp_frame._sort_frame()
                    if not lmp_frame._is_sorted()
                    else lmp_frame
                )
                lmp_frame.types = [type_info[t] for t in lmp_frame.types]

                xyz.write_frame(lmp_frame)

        except EOFError:
            ...


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
    """
    nframe = np.inf if nframe == -1 else nframe
    with reader.LAMMPS(lammpstrjtraj) as lmp:
        try:
            iframe = 0
            dframe = lmp.read_frame()
            while iframe < nframe:
                lmp_frame = lmp.read_frame()

                iframe += 1
                dframe = lmp_frame

        except EOFError:
            if nframe != np.inf:
                warnings.warn(
                    f"frame {nframe} does not exist in the trajectory file, "
                    f"therefore the last frame ({iframe}) was written."
                )

    dframe = dframe._sort_frame() if not dframe._is_sorted() else dframe
    writer.in_lammps(inlammps_name, dframe)
