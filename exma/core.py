#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Core classes and functions of exma."""

# ============================================================================
# IMPORTS
# ============================================================================

import ctypes as ct
import os
import pathlib
import sysconfig

import numpy as np

# ============================================================================
# CONSTANTS
# ============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))


# ============================================================================
# CLASSES
# ============================================================================


class AtomicSystem:
    """Define the configurations of the atomic system.

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

    def _is_sorted(self):
        """Tells if the array x is sorted (-> True) or not (-> False)."""
        return (np.diff(self.idx) >= 0).all()

    def _sort_traj(self, dontsort=("natoms", "box")):
        """Sort all the traj from the sortening of the atoms id."""
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

        self.frame = AtomicSystem()

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

    def write_frame(self):
        """Write the actual frame on the file."""
        raise NotImplementedError("Implemented in child classes.")


# ============================================================================
# FUNCTIONS
# ============================================================================


def pbc_distances(frame_c, frame_i, type_c, type_i):
    """Periodic boundary conditions distances.

    A function to compute the distances between a central and an interact
    group of atoms with the minimum image convention, in which each individual
    particle inteacts with the closest image of the copies of the system.

    Parameters
    ----------
    frame_c : `exma.core.AtomicSystem`
        which have the atomic configurations of the central atoms and must
        include `box` not None.

    frame_i : `exma.core.AtomicSystem`
        which have the atomic configurations of the interact atoms and must
        include `box` not None.

    type_c : int or str
        type of central atoms

    type_i : int or str
        type of interacting atoms

    Returns
    -------
    np.array
        array with a vector where the first natoms_i components are the
        distances of the interacting atoms to the first central one, then the
        second natoms_i to the second central atom, etc.
    """
    lib_pbc_distances = ct.CDLL(
        str(PATH / "lib" / "lib_pbc_distances")
        + sysconfig.get_config_var("EXT_SUFFIX")
    )
    distance_matrix_c = lib_pbc_distances.distance_matrix
    distance_matrix_c.argtypes = [
        ct.c_int,
        ct.c_int,
        ct.c_void_p,
        ct.c_void_p,
        ct.c_void_p,
        ct.c_void_p,
    ]

    mask_c = frame_c.types == type_c
    mask_i = frame_i.types == type_i
    natoms_c = np.count_nonzero(mask_c)
    natoms_i = np.count_nonzero(mask_i)

    box = frame_i.box
    xc, yc, zc = frame_c.x[mask_c], frame_c.y[mask_c], frame_c.z[mask_c]
    xi, yi, zi = frame_i.x[mask_i], frame_i.y[mask_i], frame_i.z[mask_i]

    distrix = np.zeros(natoms_c * natoms_i, dtype=np.float32)

    # prepare data for C code
    box = np.asarray(box, dtype=np.float32)
    box_c = box.ctypes.data_as(ct.POINTER(ct.c_void_p))

    xcentral_c = np.concatenate((xc, yc, zc)).astype(np.float32)
    xcentral_c = xcentral_c.ctypes.data_as(ct.POINTER(ct.c_void_p))

    xinteract_c = np.concatenate((xi, yi, zi)).astype(np.float32)
    xinteract_c = xinteract_c.ctypes.data_as(ct.POINTER(ct.c_void_p))

    distrix_c = distrix.ctypes.data_as(ct.POINTER(ct.c_void_p))

    # calculates the distance matrix between interact and central atoms
    distance_matrix_c(
        natoms_c, natoms_i, box_c, xcentral_c, xinteract_c, distrix_c
    )

    return distrix
