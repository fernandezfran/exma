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


class EffectiveNeighbors:
    """Emipirical way to describe charge transfer and coordination in solids.

    The empirical effective coordination model [1]_, used to calculate the
    effective neighbors, assumes that the interact atoms donate more of its
    electron to the closest central atoms. Then, fractions of the interact
    atom can be assigned to the various central neighbors atoms [2]_.

    Parameters
    ----------
    natoms : int
        number of atoms

    atom_type_central : int or str
        type of central atoms

    atom_type_interact : int or str
        type of interacting atoms

    References
    ----------
    .. [1] Hoppe, R., Voigt, S., Glaum, H., Kissel, J., Müller, H.P. and
       Bernet, K., 1988. A new route to charge distributions in ionic solids.
       `Journal of the Less Common Metals`, 156(1-2), pp.105-122.
    .. [2] Chevrier, V.L. and Dahn, J.R., 2010. First principles studies of
       disordered lithiated silicon. `Journal of the Electrochemical Society`,
       157(4), p.A392.

    """

    def __init__(self, natoms, atom_type_central, atom_type_interact):

        self.natoms = natoms
        self.atom_type_central = atom_type_central
        self.atom_type_interact = atom_type_interact

        lib_en = ct.CDLL(
            str(PATH / "lib" / "lib_en")
            + sysconfig.get_config_var("EXT_SUFFIX")
        )
        self.distance_matrix_c = lib_en.distance_matrix
        self.distance_matrix_c.argtypes = [
            ct.c_int,
            ct.c_int,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
        ]

    def of_this_frame(self, box_size, atom_type, positions):
        """Obtain the efective (interact) neighbors of the actual frame.

        Parameters
        ----------
        box_size : np.array
            the box size in x, y, z

        atom_type : np.array
            types of atoms

        positions : np.array
            the positions in the SoA convention (i.e. first all the x,
            then y and then z)

        Returns
        -------
        np.array
            effective (interact) neighbor of the central atoms in the same
            order that are in the positions vector
        """
        # calculates the distance matrix between interact and central atoms
        positions = np.split(positions, 3)
        x_c = positions[0][atom_type == self.atom_type_central]
        y_c = positions[1][atom_type == self.atom_type_central]
        z_c = positions[2][atom_type == self.atom_type_central]
        x_central = np.concatenate((x_c, y_c, z_c)).astype(np.float32)
        n_central = np.intc(len(x_central) / 3)

        x_i = positions[0][atom_type == self.atom_type_interact]
        y_i = positions[1][atom_type == self.atom_type_interact]
        z_i = positions[2][atom_type == self.atom_type_interact]
        x_interact = np.concatenate((x_i, y_i, z_i)).astype(np.float32)
        n_interact = np.intc(len(x_interact) / 3)

        distrix = np.zeros(n_central * n_interact, dtype=np.float32)
        weitrix = distrix

        box_size = box_size.astype(np.float32)
        box_c = box_size.ctypes.data_as(ct.POINTER(ct.c_void_p))

        x_c = x_central.ctypes.data_as(ct.POINTER(ct.c_void_p))
        x_i = x_interact.ctypes.data_as(ct.POINTER(ct.c_void_p))

        distrix_c = distrix.ctypes.data_as(ct.POINTER(ct.c_void_p))

        self.distance_matrix_c(
            n_central, n_interact, box_c, x_c, x_i, distrix_c
        )

        # calculate the weigth of the ith neighbor of the interact atom
        bondmin = np.min(distrix)  # the smallest bond lenght
        matrix_a = np.exp(1.0 - np.power(distrix / bondmin, 6))
        bondavg = np.sum(distrix * matrix_a) / np.sum(
            matrix_a
        )  # average bond length
        weitrix = np.exp(1.0 - np.power(distrix / bondavg, 6))

        # split the weight matrix to obtain an interact atom in every row and
        #   normalize the weigths
        weitrix = np.split(weitrix, n_interact)
        weitrix = [weitrix[i] / np.sum(weitrix[i]) for i in range(n_interact)]

        # the matrix is transpose so now we have central atoms in each row and
        #   each fraction of every interact neighbor is added to obtain the
        #   effective (interact) neighbor
        weitrix = np.transpose(weitrix)
        effnei = [np.sum(weitrix[i]) for i in range(n_central)]

        return np.array(effnei, dtype=np.float32)


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
