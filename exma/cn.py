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
import warnings

import numpy as np

from .core import _is_sorted, _sort_traj
from .io import reader

# ============================================================================
# CONSTANTS
# ============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))


# ============================================================================
# CLASSES
# ============================================================================


class CoordinationNumber:
    """Coordination Number (CN) implementation.

    The CN, also called ligancy when we are refearing to the first
    coordination shell, of a central atom is the number of atoms inside the
    empty sphere defined by an outer and an inner cut-off radius.

    Parameters
    ----------
    ftraj : str
        the string corresponding with the filename with the molecular
        dynamics trajectory

    type_c : int or str
        type of central atoms

    type_i : int or str
        type of interacting atoms

    rcut_e : float
        external cut-off radius of the shell

    rcut_i : float, default=0.0
        internal cut-off radius of the shell

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
        ftraj,
        type_c,
        type_i,
        rcut_e,
        rcut_i=0.0,
        start=0,
        stop=-1,
        step=1,
        pbc=True,
    ):
        self.ftraj = ftraj

        self.type_c = type_c
        self.type_i = type_i

        self.rcut_i = rcut_i
        self.rcut_e = rcut_e

        self.start = start
        self.stop = stop
        self.step = step

        self.pbc = pbc

    @property
    def _configure(self):
        """Configure the calculation.

        It defines parameters needed for the calculation of CN and the
        requirements of ctypes.
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
            else reader.XYZ(self.ftraj)
        )  # other xyz info can be ignored in cn calculations

        # pbc = True -> 1; False -> 0 in C code
        self.pbc = 1 if self.pbc else 0

        # frame counter
        self.ncn_ = 0

    def _configure_ctypes(self, types):
        """To calculate natoms_c_ for ctypes requirements.

        It receive frame["type"] as `types`. This is not an actually frame
        accumulation of CN.
        """
        # calculate masks, natoms_c_ and natoms_i_
        self.mask_c_ = types == self.type_c
        self.mask_i_ = types == self.type_i
        self.natoms_c_ = np.count_nonzero(self.mask_c_)
        self.natoms_i_ = np.count_nonzero(self.mask_i_)

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
        box = frame["box"]
        xc, yc, zc = (
            frame["x"][self.mask_c_],
            frame["y"][self.mask_c_],
            frame["z"][self.mask_c_],
        )
        xi, yi, zi = (
            frame["x"][self.mask_i_],
            frame["y"][self.mask_i_],
            frame["z"][self.mask_i_],
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
        imed = 0
        self._configure

        with self.traj_ as traj:
            try:
                for _ in range(self.start):
                    traj.read_frame()

                frame = traj.read_frame()

                # add the box if not in frame
                frame["box"] = box if box is not None else frame["box"]

                # sort the traj if is not sorted, xyz are sorted by default
                if "id" in frame.keys():
                    frame = (
                        _sort_traj(frame)
                        if not _is_sorted(frame["id"])
                        else frame
                    )

                self._configure_ctypes(frame["type"])

                nmed = self.stop - self.start
                while imed < nmed:
                    if imed % self.step == 0:
                        # add the box if not in frame
                        frame["box"] = box if box is not None else frame["box"]

                        # sort the traj if is not sorted
                        if "id" in frame.keys():
                            frame = (
                                _sort_traj(frame)
                                if not _is_sorted(frame["id"])
                                else frame
                            )

                        self._accumulate(frame)

                    imed += 1
                    frame = traj.read_frame()

            except EOFError:
                if self.stop != np.inf:
                    warnings.warn(
                        f"the trajectory does not read until {self.stop}"
                    )

            finally:
                self.cn_ = self._end()

        return np.mean(self.cn_), np.std(self.cn_)

    def save(self, filename="cn.dat"):
        """Write an output file.

        A one-column file where for the central atoms the corresponding
        coordination number averaged over the frames in which it was
        calculated is given and for the interacting atoms (which was not
        calculated) a nan.

        Parameters
        ----------
        filename : str, default="cn.dat"
            name of the file as str to write the output
        """
        with open(filename, "w") as fout:
            fout.write("# CN \n")
            j = 0
            for i, value in enumerate(self.mask_c_):
                if value:
                    fout.write(f"{self.cn_[j]:.6e}\n")
                    j += 1
                else:
                    fout.write(f"{np.nan} \n")
