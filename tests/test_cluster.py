#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ======================================================================
# IMPORTS
# ======================================================================

import exma.cluster

import numpy as np

# ======================================================================
# TESTS
# ======================================================================


def test_dbscan():
    """Test the dbscan cluster analyzer."""
    idref = np.array([0, 0, -1])

    box = np.array([1.0, 1.0, 1.0])
    rcut = 0.2
    types = np.array([1, 1, 1])
    xyz = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.45, 0.55, 0.0])

    result = exma.cluster.DBSCAN(rcut).of_this_frame(box, types, xyz, 1)

    np.testing.assert_array_equal(result, idref)


def test_effective_neighbors():
    """Test the calculation of effective_neighbors."""
    enref = np.array([1.0, 2.0])

    # this is a dumbell of atoms type 1 in y-direction crossed by a dumbell
    # of atoms type 2 in z-direction and an isolated atom near second
    # atom of type 1. then, the first atom of type 1 has 1 effective
    # neighbor (half of each dumbell of type 2), the same for the second
    # atom plus the isolated atom, so it has 2 effective neighbor.
    natoms = 5
    box = np.array([1.0, 1.0, 1.0])
    xyz = np.array(
        [
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.4,
            0.6,
            0.5,
            0.5,
            0.7,
            0.5,
            0.5,
            0.4,
            0.6,
            0.5,
        ]
    )
    types = np.array([1, 1, 2, 2, 2])

    result = exma.cluster.EffectiveNeighbors(natoms, 1, 2).of_this_frame(
        box, types, xyz
    )

    np.testing.assert_array_almost_equal(result, enref, 5)


def test_sro():
    """Test the amorphous parameter calculation in the rdf of a fcc crystal."""
    sroref = -0.8731494

    rdf_x = np.arange(0.025, 0.5, 0.05)
    rdf_y = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            3.478797,
            0.0,
            0.835459,
            0.0,
            1.955821,
            0.78305,
        ]
    )
    rcut = 0.375

    result = exma.cluster.sro(rdf_x, rdf_y, rcut)

    np.testing.assert_almost_equal(result, sroref)
