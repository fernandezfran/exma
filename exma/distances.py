#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Distances with Periodic Boundary Conditions."""

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
# FUNCTIONS
# ============================================================================


def pbc_distances(frame_c, frame_i, type_c="all", type_i="all"):
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

    type_c : int or str, default="all"
        type of central atoms, by default it computes the distances for all
        the atoms

    type_i : int or str, default="all"
        type of interacting atoms, by default it computes the distances for all
        the atoms

    Returns
    -------
    np.array
        array with the distance matrix where the rows are the central atoms
        and the columns the interacting ones.
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

    if type_c == "all":
        natoms_c = frame_c.natoms
        mask_c = np.full(natoms_c, True)
    else:
        mask_c = frame_c._mask_type(type_c)
        natoms_c = frame_c._natoms_type(mask_c)

    if type_i == "all":
        natoms_i = frame_i.natoms
        mask_i = np.full(natoms_i, True)
    else:
        mask_i = frame_i._mask_type(type_i)
        natoms_i = frame_i._natoms_type(mask_i)

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

    return distrix.reshape((natoms_c, natoms_i))
