#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ======================================================================
# IMPORTS
# ======================================================================

import exma.sro

import numpy as np

# ======================================================================
# TESTS
# ======================================================================


def test_amorphous():
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

    result = exma.sro.sro(rdf_x, rdf_y, rcut)

    np.testing.assert_almost_equal(result, sroref)
