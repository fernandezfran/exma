#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Implementation of Mean Square Displacement."""

# ============================================================================
# IMPORTS
# ============================================================================

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from ._mdobservable import MDObservable

# ============================================================================
# CLASSES
# ============================================================================


class MeanSquareDisplacement(MDObservable):
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

    type_e : int or str, default="all"
        the type of the element for which the msd is going to be calculated, by
        default it calculates the msd of all atoms.

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
        self, ftraj, dt, type_e="all", start=0, stop=-1, step=1, xyztype="xyz"
    ):
        super().__init__(ftraj, start, stop, step, xyztype)

        self.dt = dt
        self.type_e = type_e

    def _local_configure(self, frame):
        """Define the reference frame."""
        # mask of atoms of type e
        if self.type_e == "all":
            self.mask_e_ = np.array([True] * frame.natoms)
        else:
            self.mask_e_ = frame._mask_type(self.type_e)

        # reference positions
        self.xref_ = frame.x[self.mask_e_]
        self.yref_ = frame.y[self.mask_e_]
        self.zref_ = frame.z[self.mask_e_]
        if frame.ix is not None:
            self.xref_ = self.xref_ + frame.box[0] * frame.ix[self.mask_e_]
            self.yref_ = self.yref_ + frame.box[1] * frame.iy[self.mask_e_]
            self.zref_ = self.zref_ + frame.box[2] * frame.iz[self.mask_e_]

        self.mean_square_displacement = []

    def _accumulate(self, frame):
        """Calculate the msd of a single frame."""
        x = frame.x[self.mask_e_]
        y = frame.y[self.mask_e_]
        z = frame.z[self.mask_e_]

        if frame.ix is not None:
            x = x + frame.box[0] * frame.ix[self.mask_e_]
            y = y + frame.box[1] * frame.iy[self.mask_e_]
            z = z + frame.box[2] * frame.iz[self.mask_e_]

        x = x - self.xref_
        y = y - self.yref_
        z = z - self.zref_

        msd = np.square(x) + np.square(y) + np.square(z)

        self.mean_square_displacement.append(np.mean(msd))

    def calculate(self, box=None):
        """Calculate the MSD.

        Parameters
        ----------
        box : np.array, default=None
            the lenght of the box in each x, y, z direction, required when
            the trajectory is in an xyz file.

        Returns
        -------
        pd.DataFrame
            A `pd.DataFrame` with the time and the msd as columns.
        """
        super()._calculate(box)
        self.df_msd_ = pd.DataFrame(
            {
                "t": self.dt * np.arange(0, self.imed, self.step),
                "msd": np.array(self.mean_square_displacement),
            }
        )

        return self.df_msd_

    def plot(self, ax=None, plot_kws=None):
        """Plot the calculated MSD.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axis, default=None
            current metplotlib axis

        plot_kws : dict, defualt=None
            additional keyword arguments that are passed and are documented
            in `matplotlib.pyplot.plot`.

        Returns
        -------
        matplotlib.pyplot.Axis
            the axis with the plot
        """
        ax = plt.gca() if ax is None else ax

        plot_kws = {} if plot_kws is None else plot_kws

        ax.set_xlabel("t")
        ax.set_ylabel("msd")
        ax.plot(self.df_msd_["t"], self.df_msd_["msd"], **plot_kws)

        return ax
