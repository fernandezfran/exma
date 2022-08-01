#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Core classes of exma."""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np


# ============================================================================
# CLASSES
# ============================================================================


class AtomicSystem:
    """Define the configurations of the atomic system.

    Parameters
    ----------
    natoms : int, default=None
        the number of atoms in the atomic system

    box : np.array, default=None
        the box lenght in each direction

    types : np.array, default=None
        the types of the atoms

    idx : np.array, default=None
        the index of the atoms

    q : np.array, default=None
        a property of each atom in the atomic system

    x : np.array, default=None
        the positions of the atoms in the x direction

    y : np.array, default=None
        the positions of the atoms in the y direction

    z : np.array, default=None
        the positions of the atoms in the z direction

    ix : np.array, default=None
        the corresponding image of the positions of the atoms in the x
        direction

    iy : np.array, default=None
        the corresponding image of the positions of the atoms in the y
        direction

    iz : np.array, default=None
        the corresponding image of the positions of the atoms in the z
        direction
    """

    def __init__(
        self,
        natoms=None,
        box=None,
        types=None,
        idx=None,
        q=None,
        x=None,
        y=None,
        z=None,
        ix=None,
        iy=None,
        iz=None,
    ):
        self.natoms = natoms
        self.box = box

        self.idx = idx
        self.types = types

        self.q = q

        self.x = x
        self.y = y
        self.z = z

        self.ix = ix
        self.iy = iy
        self.iz = iz

    def _mask_type(self, atom_type):
        """Get a masked array by an specific type of atom."""
        return self.types == atom_type

    def _natoms_type(self, mask_type):
        """Count the number of atoms of an specific type."""
        return np.count_nonzero(mask_type)

    def _sorted(self):
        """Tells if the array x is sorted (-> True) or not (-> False)."""
        return (np.diff(self.idx) >= 0).all()

    def _sort(self, dontsort=("natoms", "box")):
        """Sort the Atomic System from the sortening of the atoms id."""
        id_argsort = np.argsort(self.idx)

        for key in self.__dict__.keys():
            if (
                key.startswith("_")
                or key in dontsort
                or self.__dict__[key] is None
            ):
                continue

            self.__dict__[key] = self.__dict__[key][id_argsort]

        return self

    def _unwrap(self, m=None):
        """Unwrap the Atomic System m masked outside the box."""
        m = m if m is not None else np.full(self.natoms, True)

        self.x[m] = self.x[m] + self.box[0] * self.ix[m]
        self.y[m] = self.y[m] + self.box[1] * self.iy[m]
        self.z[m] = self.z[m] + self.box[2] * self.iz[m]

        return self

    def _wrap(self, m=None):
        """Wrap the Atomic System m masked inside the box."""
        m = m if m is not None else np.full(self.natoms, True)
        indexes = np.where(m)[0]

        pos = np.zeros(3, dtype=np.float32)
        for im, x, y, z in zip(indexes, self.x[m], self.y[m], self.z[m]):

            pos[0], pos[1], pos[2] = x, y, z
            for k, kbox in enumerate(self.box):
                while pos[k] < kbox:
                    pos[k] += kbox
                while pos[k] > kbox:
                    pos[k] -= kbox

            self.x[im], self.y[im], self.z[im] = pos[0], pos[1], pos[2]

        return self


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

            # sort the frames if is not sorted, this might not be necessary
            # for all observables and all trajectories
            if frame.idx is not None:
                frame = frame._sort() if not frame._sorted() else frame

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
