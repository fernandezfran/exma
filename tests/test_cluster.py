#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ============================================================================
# IMPORTS
# ============================================================================

import exma.cluster
from exma.core import AtomicSystem

import numpy as np

# ============================================================================
# TESTS
# ============================================================================


def test_effective_neighbors():
    """Test the calculation of effective_neighbors.

    this is a dumbell of atoms type Si in y-direction crossed by a dumbell
    of atoms type Li in z-direction and an isolated atom near second
    atom of type Si. then, the first atom of type Si has Si effective
    neighbor (half of each dumbell of type Li), the same for the second
    atom plus the isolated atom, so it has Li effective neighbor.
    """
    enref = np.array([1.0, 2.0])

    frames = [
        AtomicSystem(
            natoms=5,
            types=np.array(["Si", "Si", "Li", "Li", "Li"]),
            x=np.full(5, 0.5),
            y=np.array([0.4, 0.6, 0.5, 0.5, 0.7]),
            z=np.array([0.5, 0.5, 0.4, 0.6, 0.5]),
        )
    ]
    result = exma.cluster.EffectiveNeighbors(frames, "Si", "Li").calculate(
        box=np.full(3, 1)
    )

    np.testing.assert_array_almost_equal(result, enref, 5)


def test_dbscan_id():
    """Test the dbscan cluster ids."""
    idref = np.array([0, 0, -1])

    frame = AtomicSystem(
        box=np.array([1.0, 1.0, 1.0]),
        types=np.array([1, 1, 1]),
        x=np.array([0.0, 0.0, 0.5]),
        y=np.array([0.0, 0.0, 0.0]),
        z=np.array([0.45, 0.55, 0.0]),
    )

    rcut = 0.2
    result = exma.cluster.DBSCAN(1, 1, rcut).of_this_frame(frame)

    np.testing.assert_array_equal(result, idref)


def test_dbscan_characterize():
    """Test the dbscan cluster simple characterize."""
    isolatedref = 1
    clustersref = 1

    frame = AtomicSystem(
        box=np.array([1.0, 1.0, 1.0]),
        types=np.array([1, 1, 1]),
        x=np.array([0.0, 0.0, 0.5]),
        y=np.array([0.0, 0.0, 0.0]),
        z=np.array([0.45, 0.55, 0.0]),
    )

    dbscan = exma.cluster.DBSCAN(1, 1, 0.2)
    dbscan.of_this_frame(frame)
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
