#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021-2023, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Implementation of Velocity Auto-Correlation Function."""

# ============================================================================
# IMPORTS
# ============================================================================

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from .core import MDObservable

# ============================================================================
# CLASSES
# ============================================================================


class VelocityAutocorrelationFunction(MDObservable):
    """Velocity Autocorrelation Function (VACF) implementation.

    The VACF is a measure of the correlation between the velocities of atoms
    in a molecular system as a function of time. It is calculated by
    multiplying the velocities of the atoms at a given time by the velocities
    of the same atoms at a reference time. The average over all atoms is
    considered.

    Parameters
    ----------
    frames : list
        a list with all the frames of the molecular dynamics trajectory, where
        each one is an `exma.core.AtomicSystem` and have the velocities
        defined.

    dt : int or float
        the timestep, how separated the measured frames are from each
        other, in the corresponding time units

    type_e : int or str, default="all"
        the type of the element for which the vacf is going to be calculated,
        by default it calculates the vacf of all atoms.

    start : int, default=0
        the initial frame

    stop : int, default=-1
        the last frame, by default -1 means the last

    step : int, default=1
        the incrementation if it is necessary to skip frames
    """

    def __init__(self, frames, dt, type_e="all", start=0, stop=-1, step=1):
        super().__init__(frames, start, stop, step)

        self.dt = dt
        self.type_e = type_e

    def _local_configure(self, frame):
        """Define the reference frame."""
        # mask of atoms of type e
        if self.type_e == "all":
            self.mask_e_ = np.full(frame.natoms, True)
        else:
            self.mask_e_ = frame._mask_type(self.type_e)

        # reference velocities
        self.vxref_ = frame.vx[self.mask_e_]
        self.vyref_ = frame.vy[self.mask_e_]
        self.vzref_ = frame.vz[self.mask_e_]

        self.velocity_autocorrelation_function = []

    def _accumulate(self, frame):
        """Calculate the vacf of a single frame."""
        vxsq = frame.vx[self.mask_e_] * self.vxref_
        vysq = frame.vy[self.mask_e_] * self.vyref_
        vzsq = frame.vz[self.mask_e_] * self.vzref_

        vacf = vxsq + vysq + vzsq

        self.velocity_autocorrelation_function.append(np.mean(vacf))

    def calculate(self):
        """Calculate the VACF.

        Returns
        -------
        pd.DataFrame
            A `pd.DataFrame` with the time and the vacf as columns.
        """
        super()._calculate()
        self.df_vacf_ = pd.DataFrame(
            {
                "t": self.dt * np.arange(0, len(self.frames), self.step),
                "vacf": np.array(self.velocity_autocorrelation_function),
            }
        )

        return self.df_vacf_

    def plot(self, ax=None, plot_kws=None):
        """Plot the calculated VACF.

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
        ax.set_ylabel("vacf")
        ax.plot(self.df_vacf_["t"], self.df_vacf_["vacf"], **plot_kws)

        return ax
