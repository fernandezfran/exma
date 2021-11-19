#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ======================================================================
# IMPORTS
# ======================================================================

import exma.clusterization

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

    result = exma.clusterization.DBSCAN(rcut).of_this_frame(
        box, types, xyz, 1
    )

    np.testing.assert_array_almost_equal(result[0], xyz)
    np.testing.assert_array_equal(result[1], idref)
