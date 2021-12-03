#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Core classes and functions of exma."""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np

# ============================================================================
# IO CLASSES
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


class TrajectoryWriter:
    """Class to write trajectory files.

    Parameters
    ----------
    file_traj : str
        name of the file where the trajectories in xyz format are going to
        be written

    ftype : str
        different type of files depending on the child class.
    """

    def __init__(self, file_traj, ftype):
        self.file_traj = open(file_traj, "w")
        self.ftype = ftype

    def write_frame(self):
        """Write the actual frame on the file."""
        raise NotImplementedError("Implemented in child classes.")

    def file_close(self):
        """Close the trayectory file."""
        self.file_traj.close()


# ============================================================================
# FUNCTIONS
# ============================================================================


def _is_sorted(idx):
    """Tells if the array x is sorted (-> True) or not (-> False)."""
    return (np.diff(idx) >= 0).all()


def _sort_traj(frame, dontsort=("natoms", "box")):
    """Sort all the traj from the sortening of the atoms id."""
    id_argsort = np.argsort(frame["id"])

    for key in frame.keys():
        frame[key] = (
            frame[key][id_argsort] if key not in dontsort else frame[key]
        )
    return frame
