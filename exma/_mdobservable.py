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
# CLASSES
# ============================================================================


class MDObservable:
    """Class to define the structure of the molecular dynamics observable.

    Parameters
    ----------
    frames : list
        a list with all the frames of the molecular dynamics trajectory, where
        each one is an `exma.core.AtomicSystem`.

    start : int, default=0
        the initial frame

    stop : int, default=-1
        the last frame, by default -1 means the last

    step : int, default=1
        the incrementation if it is necessary to skip frames
    """

    def __init__(self, frames, start=0, stop=-1, step=1):
        self.frames = frames

        self.start = start
        self.stop = stop
        self.step = step

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
        """Observable main loop, leave everything ready to do self._end().

        Parameters
        ----------
        box : np.array, default=None
            the lenght of the box in each x, y, z direction, required when
            the trajectory comes from an xyz file.
        """
        # select the production frames
        self.frames = (
            self.frames[self.start :: self.step]
            if self.stop == -1
            else self.frames[self.start : self.stop : self.step]
        )

        for i, frame in enumerate(self.frames):
            # add the box if not in frame
            frame.box = box if box is not None else frame.box

            # sort the traj if is not sorted, this might not be necessary for
            # all observables and all trajectories
            if frame.idx is not None:
                frame = (
                    frame._sort_frame() if not frame._is_sorted() else frame
                )

            if i == 0:
                self._local_configure(frame)

            self._accumulate(frame)

    def calculate(self):
        """Calculate the observable."""
        raise NotImplementedError("Implemented in child classes.")

    def to_dataframe(self):
        """Convert the results to pandas.DataFrame."""
        raise NotImplementedError("Implemented in child classes.")

    def plot(self):
        """Make a plot of the observable."""
        raise NotImplementedError("Implemented in child classes.")
