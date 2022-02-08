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
from ..core import _is_sorted, _sort_traj

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
    xyz = reader.XYZ(xyztraj, xyzftype)
    lmp = writer.LAMMPS(lammpstrj_name)

    with reader.XYZ(xyztraj, xyzftype) as xyz, writer.LAMMPS(
        lammpstrj_name
    ) as lmp:
        xyz_frame = xyz.read_frame()

        xyz_frame["type"] = [cell_info["type"][t] for t in xyz_frame["type"]]
        xyz_frame = {
            key: value
            for key, value in zip(xyz_frame.keys(), xyz_frame.values())
            if value is not None
        }
        cell_info["id"] = np.arange(1, xyz_frame["natoms"] + 1)
        del cell_info["type"]

        lmp.write_frame(dict(cell_info, **xyz_frame))


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

    dframe["type"] = [cell_info["type"][t] for t in dframe["type"]]
    dframe = {
        key: value
        for key, value in zip(dframe.keys(), dframe.values())
        if value is not None
    }

    newframe = dframe
    if "q" in cell_info.keys():
        keys = list(dframe.keys())
        keys.insert(2, "q")

        newframe = {}
        for k in keys:
            if k == "q":
                newframe[k] = cell_info[k]
            else:
                newframe[k] = dframe[k]
        del cell_info["q"]

    cell_info["id"] = np.arange(1, dframe["natoms"] + 1)
    del cell_info["type"]

    writer.in_lammps(inlammps_name, dict(cell_info, **newframe))


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
    with reader.LAMMPS(lammpstrjtraj) as lmp, writer.XYZ(xyz_name) as xyz:
        lmp_frame = lmp.read_frame()
        lmp_frame = (
            _sort_traj(lmp_frame)
            if not _is_sorted(lmp_frame["id"])
            else lmp_frame
        )
        lmp_frame["type"] = [type_info[t] for t in lmp_frame["type"]]

        xyz_frame = {
            key: value
            for key, value in zip(lmp_frame.keys(), lmp_frame.values())
            if key in ["natoms", "type", "x", "y", "z", "ix", "iy", "iz"]
        }

        xyz.write_frame(xyz_frame)


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

    dframe = _sort_traj(dframe) if not _is_sorted(dframe["id"]) else dframe
    writer.in_lammps(inlammps_name, dframe)
