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
    filename : str
        name of the file where the trajectories in xyz format are

    ftype : str
        different type of files depending on the child class.
    """

    def __init__(self, filename, ftype):
        self.filename = filename
        self.ftype = ftype

    def __enter__(self):
        """Use the open() method."""
        self.file_traj_ = open(self.filename, "r")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Use the close() method."""
        self.file_traj_.close()

    def read_frame(self):
        """Read the actual frame of the file."""
        raise NotImplementedError("Implemented in child classes.")


class TrajectoryWriter:
    """Class to write trajectory files.

    Parameters
    ----------
    filename : str
        name of the file where the trajectories in xyz format are going to
        be written

    ftype : str
        different type of files depending on the child class.
    """

    def __init__(self, filename, ftype):
        self.filename = filename
        self.ftype = ftype

    def __enter__(self):
        """Use the open() method."""
        self.file_traj_ = open(self.filename, "w")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Use the close() method."""
        self.file_traj_.close()

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
