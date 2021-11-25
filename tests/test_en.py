#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ======================================================================
# IMPORTS
# ======================================================================

import exma.en

import numpy as np

# ======================================================================
# TESTS
# ======================================================================


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

    result = exma.en.EffectiveNeighbors(natoms, 1, 2).of_this_frame(
        box, types, xyz
    )

    np.testing.assert_array_almost_equal(result, enref, 5)
