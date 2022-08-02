#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Radial Distribution Function Calculations."""

# ============================================================================
# IMPORTS
# ============================================================================

import ctypes as ct
import os
import pathlib
import sysconfig

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from .core import MDObservable

# ============================================================================
# CONSTANTS
# ============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))


# ============================================================================
# CLASSES
# ============================================================================


class RadialDistributionFunction(MDObservable):
    r"""Radial Distribution Function (RDF) implementation.

    The RDF is a descriptor of the variation of density of a system in
    function of distance from a reference atom. It gives the probability
    of finding an atom, relative to an ideal gas, at a given distance.

    This microscopic information gives global information about the system,
    for example, if in an RDF plot the peaks are well defined, it means that
    the system is behaving like a solid; on the other hand, if the peaks are
    broadened and decrease in intensity as the distance increases, tending
    to oscillate around 1, it means that the system behaves like a liquid;
    a straight line at 1 is an ideal gas.

    Parameters
    ----------
    frames : list
        a list with all the frames of the molecular dynamics trajectory, where
        each one is an `exma.core.AtomicSystem`.

    type_c : int or str, default="all"
        type of central atoms, by default it computes the total rdf

    type_i : int or str, default="all"
        type of interacting atoms, by default it computes the total rdf

    start : int, default=0
        the initial frame

    stop : int, default=-1
        the last frame, by default -1 means the last

    step : int, default=1
        the incrementation if it is necessary to skip frames

    rmax : float, default=10.0
        the maximum distance at which to calculate g(r), should not be
        greater than half of the shortest lenght of the box if pbc are
        considered

    nbin : int, default=100
        number of bins in the histogram

    pbc : bool, default=True
        True if periodic boundary conditions must be considered, False if
        not.

    Notes
    -----
    The definition of `rmax` and `nbin` defines the `dr`, the width of the
    histogram as

    .. math::
        dr = \frac{rmax}{nbin}

    for example, the default values give `dr=0.1`.
    """

    def __init__(
        self,
        frames,
        type_c="all",
        type_i="all",
        start=0,
        stop=-1,
        step=1,
        rmax=10.0,
        nbin=100,
        pbc=True,
    ):
        super().__init__(frames, start, stop, step)

        self.type_c = type_c
        self.type_i = type_i

        self.rmax = rmax
        self.nbin = nbin
        self.pbc = pbc

        # the width of the bar in the histogram
        self.dgofr_ = self.rmax / self.nbin

    def _local_configure(self, frame):
        """Configure the Radial Distribution Function calculus.

        It receive a frame and define the boundary conditions, initializate the
        volumen, the counter and the ctypes requirements.
        """
        # pbc = True -> 1; False -> 0 in C code
        self.pbc = 1 if self.pbc else 0
        if self.pbc and frame.box is None:
            raise RuntimeError("box not defined in pbc calculation.")

        # init volume
        self.volume_ = 0.0

        # g(r) the counter
        self.ngofr_ = 0

        # calculate natoms_c_ and natoms_i_
        if self.type_c == "all" or self.type_i == "all":
            self.mask_c_ = self.mask_i_ = np.full(frame.natoms, True)
            self.natoms_c_ = self.natoms_i_ = frame.natoms
        else:
            self.mask_c_ = frame._mask_type(self.type_c)
            self.mask_i_ = frame._mask_type(self.type_i)
            self.natoms_c_ = frame._natoms_type(self.mask_c_)
            self.natoms_i_ = frame._natoms_type(self.mask_i_)

        # ctypes requirements to interact with C code
        lib_rdf = ct.CDLL(
            str(PATH / "lib" / "lib_rdf")
            + sysconfig.get_config_var("EXT_SUFFIX")
        )
        self.rdf_c_ = lib_rdf.rdf_accumulate
        self.rdf_c_.argtypes = [
            ct.c_int,
            ct.c_int,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_int,
            ct.c_float,
            ct.c_float,
            ct.c_int,
            ct.c_void_p,
        ]
        self.gofr_c_ = (ct.c_int * self.nbin)()

    def _fake_box(self, frame):
        """Box definition for clusters that do not have a defined one.

        this prevents C from being passed a pointer to an array that is
        expected to have three components and gives an array of a single nan
        for box = None.
        """
        xsize = np.max(frame.x) - np.min(frame.x)
        ysize = np.max(frame.y) - np.min(frame.y)
        zsize = np.max(frame.z) - np.min(frame.z)
        return np.array([xsize, ysize, zsize])

    def _accumulate(self, frame):
        """Accumulates the info of each frame."""
        if self.pbc:
            box = frame.box
            self.volume_ += np.prod(box)
        else:
            box = self._fake_box(frame)
            self.volume_ = 4.0 * np.pi * np.power(0.5 * np.mean(box), 3) / 3.0

        xc, yc, zc = (
            frame.x[self.mask_c_],
            frame.y[self.mask_c_],
            frame.z[self.mask_c_],
        )
        xi, yi, zi = (
            frame.x[self.mask_i_],
            frame.y[self.mask_i_],
            frame.z[self.mask_i_],
        )

        # accomodate the data type of pointers to C code
        box = np.asarray(box, dtype=np.float32)
        box_c = box.ctypes.data_as(ct.POINTER(ct.c_void_p))

        xcentral_c = np.concatenate((xc, yc, zc)).astype(np.float32)
        xcentral_c = xcentral_c.ctypes.data_as(ct.POINTER(ct.c_void_p))

        xinteract_c = np.concatenate((xi, yi, zi)).astype(np.float32)
        xinteract_c = xinteract_c.ctypes.data_as(ct.POINTER(ct.c_void_p))

        # run rdf C accumulation
        self.ngofr_ += 1
        self.rdf_c_(
            self.natoms_c_,
            self.natoms_i_,
            box_c,
            xcentral_c,
            xinteract_c,
            self.pbc,
            self.dgofr_,
            self.rmax,
            self.nbin,
            self.gofr_c_,
        )

    def _end(self):
        """Finish the calculation and normalize the data."""
        r = np.arange(
            self.dgofr_ / 2, self.rmax + self.dgofr_ / 2, self.dgofr_
        )
        gofr = np.asarray(
            np.frombuffer(self.gofr_c_, dtype=np.intc, count=self.nbin)
        )

        volume = self.volume_ / self.ngofr_
        rho = self.natoms_c_ * self.natoms_i_ / volume

        shell_vols = np.power(self.dgofr_, 3) * np.diff(
            np.power(range(self.nbin + 1), 3)
        )
        ideal = 4.0 * np.pi * shell_vols * rho / 3.0

        rdf = gofr / (self.ngofr_ * ideal)

        return r, rdf

    def calculate(self, box=None):
        """Calculate the RDF.

        Parameters
        ----------
        box : np.array, default=None
            the lenght of the box in each x, y, z direction, required when
            the trajectory is in an xyz file.

        Returns
        -------
        pd.DataFrame
            A `pd.DataFrame` with r and g(r) as columns.
        """
        super()._calculate(box)

        r, rdf = self._end()
        self.df_rdf_ = pd.DataFrame({"r": r, "rdf": rdf})

        return self.df_rdf_

    def plot(self, ax=None, plot_kws=None):
        """Plot the calculated RDF.

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

        ax.set_xlabel("r")
        ax.set_ylabel("g(r)")
        ax.plot(self.df_rdf_["r"], self.df_rdf_["rdf"], **plot_kws)

        return ax
