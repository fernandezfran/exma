#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ======================================================================
# DOCS
# ======================================================================

"""Implementation of clustering."""

# ======================================================================
# IMPORTS
# ======================================================================

import ctypes as ct
import os
import pathlib
import sysconfig

import numpy as np

import scipy.integrate

import sklearn.cluster

# =============================================================================
# CONSTANTS
# =============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))


# ============================================================================
# CLASSES
# ============================================================================


class ClusterCaracterization:

    def __init___(self):
        pass

    def _accumulate(self):
        """accumulate simple characterize()"""
        pass

    def _end(self):
        pass

    def calculate(self):
        pass

    def save(self):
        pass

    def plot(self):
        pass


class DBSCAN:
    """DBSCAN clustering using PBC for the distance matrix calculations.

    Parameters
    ----------
    eps : float
        a cutoff radius at which an atom is no longer considered part of the
        cluster.

    min_samples : int, default=2
        number of atoms required to form a cluster.

    Notes
    -----
    If the system is not under periodic boundary conditions, then
    everything can be calculated from `sklearn.cluster.DBSCAN` using
    `metrics="euclidean"`.
    """

    def __init__(self, eps, min_samples=2):
        self.eps = eps
        self.min_samples = min_samples

        lib_cluster = ct.CDLL(
            str(PATH / "lib" / "lib_cluster")
            + sysconfig.get_config_var("EXT_SUFFIX")
        )
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
            the positions in the SoA convention (i.e. first all the x,
            then y and then z).

        atom_type_c : int or str
            type of atom to which you want to perform the cluster analysis

        **kwargs
            additional keyword arguments that are passed and are documented
            in `sklearn.cluster.DBSCAN`

        Returns
        -------
        id_cluster : np.array
            as explained in `sklearn.cluster.DBSCAN`. It contains the id
            number of the cluster to which belongs the corresponding atom
            (the array is sorted). A value of -1 means that the atom is
            isolated.
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

        self.id_cluster_ = np.asarray(db.labels_, dtype=np.intc)

        return self.id_cluster_

    def characterize(self):
        """Characterization in number of clusters and isolated.

        Returns
        -------
        tuple
            with the number of clusters and the number of isolated atoms.
        """
        isolated = np.count_nonzero(self.id_cluster_ == -1)

        uniques = np.unique(self.id_cluster_).size
        clusters = uniques if isolated == 0 else uniques - 1

        return (clusters, isolated)


# ============================================================================
# FUNCTIONS
# ============================================================================


def sro(rdf_x, rdf_y, rcut, **kwargs):
    """Short range order parameter.

    To characterize the short range ordering of amorphous structures, defined
    in this work [3]_, using the itegration of the
    radial distribution function for diatomic systems.

    This parameter indicates complete randomness if it is equal to zero,
    preference for unlike neighbors if it is less than zero, and preference
    for similar neighbors (clustering) if is greater than zero.

    Parameters
    ----------
    rdf_x : np.array
        x of the radial distribution function

    rdf_y : np.array
        y of the radial distribution function

    rcut : float
        cutoff radius

    **kwargs
        Additional keyword arguments that are passed and are documented in
        `scipy.integrate.simpson`.

    Returns
    -------
    float
        amorphous short range order parameter

    References
    ----------
    .. [3] Fernandez, F., Paz, S.A., Otero, M., Barraco, D. and Leiva, E.P.,
       2021. Characterization of amorphous Li x Si structures from ReaxFF via
       accelerated exploration of local minima. `Physical Chemistry Chemical
       Physics`, 23(31), pp.16776-16784.
    """
    vol = (4.0 / 3.0) * np.pi * np.power(rcut, 3)

    mask = rdf_x < rcut
    ix = rdf_x[mask]
    iy = 4.0 * np.pi * ix * ix * rdf_y[mask]

    cab = scipy.integrate.simpson(iy, x=ix, **kwargs)

    return np.log(cab / vol)
