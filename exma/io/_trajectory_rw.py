#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Trajectory Reader and Writer superclasses."""


# ============================================================================
# CLASSES
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

    def read_traj(self):
        """Read all the trajectory of the file."""
        traj = []
        try:
            while True:
                traj.append(self.read_frame())
        except (EOFError, NotImplementedError):
            ...

        return traj


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

    def write_frame(self, frame):
        """Write the actual frame on the file."""
        raise NotImplementedError("Implemented in child classes.")

    def write_traj(self, frames):
        """Write all frames of the trajectory to a file."""
        for frame in frames:
            self.write_frame(frame)
