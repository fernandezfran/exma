#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ======================================================================
# DOCS
# ======================================================================

"""Implementation of classification through DBSCAN."""

# ======================================================================
# IMPORTS
# ======================================================================

import ctypes as ct
import os
import pathlib
import sysconfig

import numpy as np

import sklearn.cluster

# =============================================================================
# CONSTANTS
# =============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

lib_cluster = ct.CDLL(
    str(PATH / "lib" / "lib_cluster") + sysconfig.get_config_var("EXT_SUFFIX")
)

# ============================================================================
# CLASSES
# ============================================================================


class DBSCAN:
    """DBSCAN clustering using PBC for the distance matrix calculations.

    Note that if the system is not under periodic boundary conditions, then
    everything can be calculated from `sklearn.cluster.DBSCAN` using
    `metrics="euclidean"`.

    Parameters
    ----------
    eps : float
        like an rcut where an atoms stop to be considered part of a cluster

    min_samples : int (default=2)
        the number of atoms that can be a core point

    """

    def __init__(self, eps, min_samples=2):
        self.eps = eps
        self.min_samples = min_samples

        self.distance_matrix_c = lib_cluster.distance_matrix
        self.distance_matrix_c.argtypes = [
            ct.c_int,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
        ]

    def of_this_frame(
        self, box_size, atom_type, positions, atom_type_c, **kwargs
    ):
        """Obtain the labels of the DBSCAN clustering the actual frame.

        Parameters
        ----------
        box_size : np.array
            the box size in x, y, z

        atom_type : np.array
            type of atoms

        positions : np.array
            the positions in the SoA convention (i.e. first all the x, then y
            and then z).

        atom_type_c : int or str
            type of atom to which you want to perform the cluster analysis

        **kwargs
            additional keyword arguments that are passed and are documented
            in ``sklearn.cluster.DBSCAN``

        Returns
        -------
        id_cluster : np.array
            as explained in ``sklearn.cluster.DBSCAN``. It contains the id
            number of the cluster to which belongs the corresponding atom (the
            array is sorted). A value of -1 means that the atom is isolated.
        """
        positions = np.asarray(positions, dtype=np.float32)
        xyz = np.split(positions, 3)
        x, y, z = xyz[0], xyz[1], xyz[2]
        x, y, z = (
            x[atom_type == atom_type_c],
            y[atom_type == atom_type_c],
            z[atom_type == atom_type_c],
        )

        positions_c = np.concatenate((x, y, z)).astype(np.float32)
        natoms_c = np.intc(len(x))
        distrix = np.zeros(natoms_c * natoms_c, dtype=np.float32)

        # prepare data to C function
        box_size = box_size.astype(np.float32)
        box_c = box_size.ctypes.data_as(ct.POINTER(ct.c_void_p))

        x_c = positions_c.ctypes.data_as(ct.POINTER(ct.c_void_p))

        distrix_c = distrix.ctypes.data_as(ct.POINTER(ct.c_void_p))

        # a void function that modifies the values of distrix
        self.distance_matrix_c(natoms_c, box_c, x_c, distrix_c)

        distrix = distrix.reshape((natoms_c, natoms_c))
        db = sklearn.cluster.DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric="precomputed",
            **kwargs,
        ).fit(distrix)

        id_cluster = np.asarray(db.labels_, dtype=np.intc)

        return id_cluster
