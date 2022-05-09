#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Molecular Dynamics Observable superclass."""

# ============================================================================
# IMPORTS
# ============================================================================

import warnings

import numpy as np

from .io import reader

# ============================================================================
# CLASSES
# ============================================================================


class MDObservable:
    """Class to define the structure of the molecular dynamics observable.

    Parameters
    ----------
    ftraj : str
        the string corresponding with the filename with the molecular
        dynamics trajectory

    start : int, default=0
        the initial frame

    stop : int, default=-1
        the last frame, by default -1 means the last

    step : int, default=1
        the incrementation if it is necessary to skip frames

    xyztype : str, default="xyz"
        the type of the xyzfile.
    """

    def __init__(self, ftraj, start=0, stop=-1, step=1, xyztype="xyz"):
        self.ftraj = ftraj

        self.start = start
        self.stop = stop
        self.step = step

        self.xyztype = xyztype

    def _configure_reader(self):
        """Configure the reader.

        It defines the trajectory reader type (LAMMPS or XYZ) depending on
        the extension (or raises a ValueError) and define the last frame.
        """
        # configure the frame at which stop the calculation
        self.stop = np.inf if self.stop == -1 else self.stop

        # define the trajectory reader
        fextension = self.ftraj.split(".")[-1]
        if fextension not in ["lammpstrj", "xyz"]:
            raise ValueError(
                "The file must have the extension .xyz or .lammpstrj"
            )

        self.traj_ = (
            reader.LAMMPS(self.ftraj)
            if fextension == "lammpstrj"
            else reader.XYZ(self.ftraj, self.xyztype)
        )

    def _local_configure(self, frame):
        """Specific configuration of each observable."""
        raise NotImplementedError("Implemented in child classes.")

    def _accumulate(self, frame):
        """Accumulate the data of the frame."""
        raise NotImplementedError("Implemented in child classes.")

    def _end(self, frame):
        """Finish the calculation and normilize the data."""
        raise NotImplementedError("Implemented in child classes.")

    def _calculate(self, box=None):
        """Observable main loop.

        Parameters
        ----------
        box : np.array, default=None
            the lenght of the box in each x, y, z direction, required when
            the trajectory is in an xyz file.

        Returns
        -------
        None
            but leave everything ready to be able to do self._end()
        """
        self.imed = 0
        self._configure_reader()

        with self.traj_ as traj:
            try:
                for _ in range(self.start):
                    traj.read_frame()

                frame = traj.read_frame()

                # add the box if not in frame
                frame.box = box if box is not None else frame.box

                # sort the traj if is not sorted, this might not be necessary
                # for all observables.
                if frame.idx is not None:
                    frame = (
                        frame._sort_frame()
                        if not frame._is_sorted()
                        else frame
                    )

                self._local_configure(frame)

                nmed = self.stop - self.start
                while self.imed < nmed:
                    if self.imed % self.step == 0:
                        frame.box = box if box is not None else frame.box

                        if frame.idx is not None:
                            frame = (
                                frame._sort_frame()
                                if not frame._is_sorted()
                                else frame
                            )

                        self._accumulate(frame)

                    self.imed += 1
                    frame = traj.read_frame()

            except EOFError:
                if self.stop != np.inf:
                    warnings.warn(
                        f"the trajectory does not read until {self.stop}"
                    )

            finally:
                return None

    def calculate(self):
        """Calculate the observable."""
        raise NotImplementedError("Implemented in child classes.")

    def to_dataframe(self):
        """Convert the results to pandas.DataFrame."""
        raise NotImplementedError("Implemented in child classes.")

    def plot(self):
        """Make a plot of the observable."""
        raise NotImplementedError("Implemented in child classes.")
