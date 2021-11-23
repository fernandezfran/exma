#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ======================================================================
# DOCS
# ======================================================================

"""Effective Neighbors Calculations."""

# ======================================================================
# IMPORTS
# ======================================================================

import ctypes as ct
import os
import pathlib
import sysconfig

import numpy as np

# =============================================================================
# CONSTANTS
# =============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

lib_en = ct.CDLL(
    str(PATH / "lib" / "lib_en") + sysconfig.get_config_var("EXT_SUFFIX")
)

# ============================================================================
# CLASSES
# ============================================================================


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
