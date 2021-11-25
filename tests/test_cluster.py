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


def test_dbscan_id():
    """Test the dbscan cluster ids."""
    idref = np.array([0, 0, -1])

    box = np.array([1.0, 1.0, 1.0])
    rcut = 0.2
    types = np.array([1, 1, 1])
    xyz = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.45, 0.55, 0.0])

    result = exma.cluster.DBSCAN(rcut).of_this_frame(box, types, xyz, 1)

    np.testing.assert_array_equal(result, idref)


def test_dbscan_characterize():
    """Test the dbscan cluster simple characterize."""
    isolatedref = 1
    clustersref = 1

    box = np.array([1.0, 1.0, 1.0])
    rcut = 0.2
    types = np.array([1, 1, 1])
    xyz = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.45, 0.55, 0.0])

    dbscan = exma.cluster.DBSCAN(rcut)
    dbscan.of_this_frame(box, types, xyz, 1)
    isolated, clusters = dbscan.characterize()

    assert isolated == isolatedref
    assert clusters == clustersref


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
