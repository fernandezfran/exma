#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ======================================================================
# IMPORTS
# ======================================================================

import os
import pathlib

import exma.io.reader

import numpy as np

# ============================================================================
# CONSTANTS
# ============================================================================

TEST_DATA_PATH = pathlib.Path(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_data")
)


# ======================================================================
# TESTS
# ======================================================================


def test_read_log_lammps():
    """Test the read of log.lammps"""
    log = exma.io.reader.read_log_lammps(
        logname=TEST_DATA_PATH / "log.test_ref"
    )

    np.testing.assert_array_almost_equal(log["Step"], np.arange(0, 11))
    np.testing.assert_almost_equal(np.mean(log["Press"]), 14677.928, decimal=3)
    np.testing.assert_almost_equal(
        np.mean(log["PotEng"]), -6847.817, decimal=3
    )
    np.testing.assert_almost_equal(
        np.mean(log["TotEng"]), -6847.817, decimal=3
    )
    np.testing.assert_almost_equal(
        np.mean(log["Volume"]), 1310.3907, decimal=4
    )
