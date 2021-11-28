#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ======================================================================
# DOCS
# ======================================================================

"""Implementation of Mean Square Displacement."""

# ======================================================================
# IMPORTS
# ======================================================================

import warnings

import numpy as np

import pandas as pd

from . import _traj_sorter
from .io import reader

# ======================================================================
# CLASSES
# ======================================================================


class MeanSquareDisplacement:
    """Mean Square Displacement (MSD) implementation.

    The MSD is a measure of the deviation of the position of an atom
    with respect to its reference position over time. At each time value
    `t`, the MSD is defined as an ensemble average over all atoms to be
    considered.

    Parameters
    ----------
    ftraj : str
        the string corresponding with the filename with the molecular
        dynamics trajectory

    dt : int or float
        the timestep, how separated the measured frames are from each
        other, in the corresponding time units

    type_e : int or str
        the type of the element for which the msd is going to be calculated

    start : int, default=0
        the initial frame

    stop : int, default=-1
        the last frame, by default -1 means the last

    step : int, default=1
        the incrementation if it is necessary to skip frames

    xyztype : str, default="xyz"
        the string that describes the type of xyz file, to pass to reader, it
        is only necessary if ftraj ends with .xyz extension

    Notes
    -----
    The trajectory must be unwrapped outside the simulation cell, if it
    is wrapped, the image to which it corresponds each atom must be found
    in the trajectory file.
    """

    def __init__(
        self, ftraj, dt, type_e, start=0, stop=-1, step=1, xyztype="xyz"
    ):
        self.ftraj = ftraj
        self.xyztype = xyztype

        self.dt = dt
        self.type_e = type_e

        self.start = start
        self.stop = stop
        self.step = step

    @property
    def _configure(self):
        """Configure the calculation.

        It defines the trajectory reader type (LAMMPS or XYZ) depending on
        the extension or raises a ValueError.
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

    def _reference_frame(self, frame):
        """Define the reference frame."""
        # mask of atoms of type e
        self.mask_e_ = frame["type"] == self.type_e

        # reference positions
        self.xref_ = frame["x"][self.mask_e_]
        self.yref_ = frame["y"][self.mask_e_]
        self.zref_ = frame["z"][self.mask_e_]
        if "ix" in frame.keys():
            self.xref_ = (
                self.xref_ + frame["box"][0] * frame["ix"][self.mask_e_]
            )
            self.yref_ = (
                self.yref_ + frame["box"][1] * frame["iy"][self.mask_e_]
            )
            self.zref_ = (
                self.zref_ + frame["box"][2] * frame["iz"][self.mask_e_]
            )

    def _on_this_frame(self, frame):
        """Calculate the msd of a single frame."""
        x = frame["x"][self.mask_e_]
        y = frame["y"][self.mask_e_]
        z = frame["z"][self.mask_e_]

        if "ix" in frame.keys():
            x = x + frame["box"][0] * frame["ix"][self.mask_e_]
            y = y + frame["box"][1] * frame["iy"][self.mask_e_]
            z = z + frame["box"][2] * frame["iz"][self.mask_e_]

        x = x - self.xref_
        y = y - self.yref_
        z = z - self.zref_

        msd = x * x + y * y + z * z
        return np.mean(msd)  # np.sum(msd) / self.natoms_e_

    def calculate(self, box=None):
        """Calculate the MSD.

        Parameters
        ----------
        box : np.array
            the lenght of the box in each x, y, z direction, required when
            the trajectory is in an xyz file.

        Returns
        -------
        pd.DataFrame
            a dataframe with the time and the msd as columns.
        """
        imed = 0
        self._configure

        mean_square_displacement = []
        try:
            for _ in range(self.start):
                self.traj_.read_frame()

            frame = self.traj_.read_frame()

            # add the box if not in frame
            frame["box"] = box if box is not None else frame["box"]

            # sort the traj if is not sorted, xyz are sorted by construction
            if "id" in frame.keys():
                frame = (
                    _traj_sorter._sort_traj(frame)
                    if not _traj_sorter._is_sorted(frame["id"])
                    else frame
                )

            self._reference_frame(frame)

            while imed < self.stop / self.step:
                if imed % self.step == 0:
                    # add the box if not in frame
                    frame["box"] = box if box is not None else frame["box"]

                    # sort the traj if is not sorted
                    if "id" in frame.keys():
                        frame = (
                            _traj_sorter._sort_traj(frame)
                            if not _traj_sorter._is_sorted(frame["id"])
                            else frame
                        )

                    mean_square_displacement.append(self._on_this_frame(frame))
                    imed += 1

                frame = self.traj_.read_frame()

        except EOFError:
            if self.stop != np.inf:
                warnings.warn(
                    f"the trajectory does not read until {self.stop}"
                )

        finally:
            self.traj_.file_close()

            self.df_msd_ = pd.DataFrame(
                {
                    "t": self.dt * np.arange(0, imed),
                    "msd": np.array(mean_square_displacement),
                }
            )

            return self.df_msd_

    def save(self):
        """To be implemented soon."""
        raise NotImplementedError("To be implemented soon.")

    def plot(self):
        """To be implemented soon."""
        raise NotImplementedError("To be implemented soon.")
