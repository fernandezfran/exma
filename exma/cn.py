#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Coordination Number Calculations."""

# ============================================================================
# IMPORTS
# ============================================================================

import ctypes as ct
import os
import pathlib
import sysconfig

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


class CoordinationNumber(MDObservable):
    """Coordination Number (CN) implementation.

    The CN, also called ligancy when we are refearing to the first
    coordination shell, of a central atom is the number of atoms inside the
    empty sphere defined by an outer and an inner cut-off radius.

    Parameters
    ----------
    frames : list
        a list with all the frames of the molecular dynamics trajectory, where
        each one is an `exma.core.AtomicSystem`.

    rcut_e : float
        external cut-off radius of the shell

    rcut_i : float, default=0.0
        internal cut-off radius of the shell

    type_c : int or str, default="all"
        type of central atoms, by default it computes the total cn

    type_i : int or str, default="all"
        type of interacting atoms, by default it computes the total cn

    start : int, default=0
        the initial frame

    stop : int, default=-1
        the last frame, by default -1 means the last

    step : int, default=1
        the incrementation if it is necessary to skip frames

    pbc : bool, default=True
        True if periodic boundary conditions must be considered, False if
        not.
    """

    def __init__(
        self,
        frames,
        rcut_e,
        rcut_i=0.0,
        type_c="all",
        type_i="all",
        start=0,
        stop=-1,
        step=1,
        pbc=True,
    ):
        super().__init__(frames, start, stop, step)

        self.type_c = type_c
        self.type_i = type_i

        self.rcut_i = rcut_i
        self.rcut_e = rcut_e

        self.pbc = pbc

    def _local_configure(self, frame):
        """Configure the Coordination Number calculus.

        It receive a frame and define the boundary conditions, sets the counter
        and the ctypes requirements.
        """
        # pbc = True -> 1; False -> 0 in C code
        self.pbc = 1 if self.pbc else 0

        # frame counter
        self.ncn_ = 0

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
        lib_cn = ct.CDLL(
            str(PATH / "lib" / "lib_cn")
            + sysconfig.get_config_var("EXT_SUFFIX")
        )
        self.cn_c_ = lib_cn.cn_accumulate
        self.cn_c_.argtypes = [
            ct.c_int,
            ct.c_int,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_int,
            ct.c_float,
            ct.c_float,
            ct.c_void_p,
        ]
        self.cn_res = (ct.c_int * self.natoms_c_)()

    def _accumulate(self, frame):
        """Accumulates the info of each frame."""
        box = frame.box
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

        # run th cn C accumulation
        self.ncn_ += 1
        self.cn_c_(
            self.natoms_c_,
            self.natoms_i_,
            box_c,
            xcentral_c,
            xinteract_c,
            self.pbc,
            self.rcut_i,
            self.rcut_e,
            self.cn_res,
        )

    def _end(self):
        """Finish the calculation and normalize the data."""
        cn = np.asarray(
            np.frombuffer(self.cn_res, dtype=np.intc, count=self.natoms_c_)
        )
        return cn / self.ncn_

    def calculate(self, box=None):
        """Calculate the CN.

        Parameters
        ----------
        box : np.array, default=None
            the lenght of the box in each x, y, z direction, required when
            the trajectory is in an xyz file.

        Returns
        -------
        tuple
            a tuple with the average cn number and its standard deviation.
        """
        super()._calculate(box)
        self.cn_ = self._end()

        return np.mean(self.cn_), np.std(self.cn_)

    def to_dataframe(self):
        """Convert the results to a pandas.DataFrame.

        A one-column DataFrame where for the central atoms the corresponding
        coordination number averaged over the frames in which it was
        calculated is given and for the interacting atoms (which was not
        calculated) a np.nan.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with the coordination number data in the column "cn".
        """
        j = 0
        cn = []
        for _, value in enumerate(self.mask_c_):
            if value:
                cn.append(self.cn_[j])
                j += 1
            else:
                cn.append(np.nan)

        dict_ = {"cn": np.asarray(cn)}

        return pd.DataFrame(dict_)
