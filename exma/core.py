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
        self.file_traj = file_traj
        self.ftype = ftype

    def __enter__(self):
        """Use the open() method."""
        self.file_traj = open(self.file_traj, "r")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Use the close() method."""
        self.file_traj.close()

    def read_frame(self):
        """Read the actual frame of the file."""
        raise NotImplementedError("Implemented in child classes.")


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
        self.file_traj = file_traj
        self.ftype = ftype

    def __enter__(self):
        """Use the open() method."""
        self.file_traj = open(self.file_traj, "w")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Use the close() method."""
        self.file_traj.close()

    def write_frame(self):
        """Write the actual frame on the file."""
        raise NotImplementedError("Implemented in child classes.")


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
