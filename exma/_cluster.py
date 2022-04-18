#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Implementations of clustering.

The classes of these modules are not complete because they are under
development.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import pathlib

import numpy as np

import scipy.integrate

import sklearn.cluster

from .distances import pbc as pbc_distances

# ============================================================================
# CONSTANTS
# ============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))


# ============================================================================
# CLASSES
# ============================================================================


class EffectiveNeighbors:
    """Emipirical way to describe charge transfer and coordination in solids.

    The empirical effective coordination model [2]_, used to calculate the
    effective neighbors, assumes that the interact atoms donate more of its
    electron to the closest central atoms. Then, fractions of the interact
    atom can be assigned to the various central neighbors atoms [3]_.

    Parameters
    ----------
    type_c : int or str
        type of central atoms

    type_i : int or str
        type of interacting atoms

    References
    ----------
    .. [2] Hoppe, R., Voigt, S., Glaum, H., Kissel, J., Müller, H.P. and
       Bernet, K., 1988. A new route to charge distributions in ionic solids.
       `Journal of the Less Common Metals`, 156(1-2), pp.105-122.
    .. [3] Chevrier, V.L. and Dahn, J.R., 2010. First principles studies of
       disordered lithiated silicon. `Journal of the Electrochemical Society`,
       157(4), p.A392.

    """

    def __init__(self, type_c, type_i):
        self.type_c = type_c
        self.type_i = type_i

    def of_this_frame(self, frame):
        """Obtain the efective (interact) neighbors of the actual frame.

        Parameters
        ----------
        frame : dict
            dictionary with the keys `natoms`, `type`, `box`, `x`, `y`, `z`

        Returns
        -------
        np.array
            effective (interact) neighbor of the central atoms in the same
            order that are in the positions vector
        """
        natoms_c = np.count_nonzero(frame["type"] == self.type_c)
        natoms_i = np.count_nonzero(frame["type"] == self.type_i)

        distrix = pbc_distances(frame, frame, self.type_c, self.type_i)

        # calculate the weigth of the ith neighbor of the interact atom
        bondmin = np.min(distrix)
        matrix_a = np.exp(1.0 - np.power(distrix / bondmin, 6))
        bondavg = np.sum(distrix * matrix_a) / np.sum(matrix_a)

        weitrix = np.exp(1.0 - np.power(distrix / bondavg, 6))

        # reshape the weight matrix and transpose it to obtain an interact
        # atom in every row and normalize their weights
        weitrix = np.reshape(weitrix, (natoms_c, natoms_i)).T
        weitrix = [weitrix[i] / np.sum(weitrix[i]) for i in range(natoms_i)]

        # the matrix is transpose again so now we have central atoms in each
        # row an each fraction of every interact neighbor is added to obtain
        # the effective (interact) neighbor
        weitrix = np.transpose(weitrix)
        effnei = [np.sum(weitrix[i]) for i in range(natoms_c)]

        return np.array(effnei, dtype=np.float32)


class DBSCAN:
    """DBSCAN clustering using PBC for the distance matrix calculations.

    Parameters
    ----------
    type_c : int or str
        type of central atoms

    type_i : int or str
        type of interacting atoms

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

    def __init__(self, type_c, type_i, eps, min_samples=2):
        self.type_c = type_c
        self.type_i = type_i
        self.eps = eps
        self.min_samples = min_samples

    def of_this_frame(self, frame, **kwargs):
        """Obtain the labels of the DBSCAN clustering the actual frame.

        Parameters
        ----------
        frame : dict
            dictionary with the keys `natoms`, `type`, `box`, `x`, `y`, `z`

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
        natoms_c = np.count_nonzero(frame["type"] == self.type_c)
        natoms_i = np.count_nonzero(frame["type"] == self.type_i)

        distrix = pbc_distances(frame, frame, self.type_c, self.type_i)

        distrix = distrix.reshape((natoms_c, natoms_i))
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

        return clusters, isolated


# ============================================================================
# FUNCTIONS
# ============================================================================


def sro(rdf_x, rdf_y, rcut, **kwargs):
    """Short range order parameter.

    To characterize the short range ordering of amorphous structures, defined
    in this work [4]_, using the itegration of the radial distribution
    function for diatomic systems.

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
    .. [4] Fernandez, F., Paz, S.A., Otero, M., Barraco, D. and Leiva, E.P.,
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