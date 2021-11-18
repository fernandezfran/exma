#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ======================================================================
# IMPORTS
# ======================================================================

import exma.statistics

import numpy as np

# ======================================================================
# TESTS
# ======================================================================


def test_block_average():
    """Test the estimation of an error."""
    result = exma.statistics.block_average(
        [3.14, 3.15, 3.13, 3.13, 3.15, 3.15, 3.16, 3.12]
    ).estimate_error()

    np.testing.assert_array_equal(result[0], np.array([0, 1]))
    np.testing.assert_array_equal(result[1], np.array([8, 4]))
    np.testing.assert_array_almost_equal(
        result[2], np.array([3.1412501, 3.1412501])
    )
    np.testing.assert_array_almost_equal(
        result[3], np.array([2.299121e-05, 2.656272e-05])
    )
    np.testing.assert_array_almost_equal(
        result[4], np.array([1.228932e-05, 2.168837e-05])
    )
