#!/usr/bin/envpython
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021-2023, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ============================================================================
# IMPORTS
# ============================================================================

import os
import pathlib

import exma.cn
from exma import read_xyz

import numpy as np

import pandas as pd

import pytest

# ============================================================================
# CONSTANTS
# ============================================================================

TEST_DATA_PATH = pathlib.Path(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_data")
)

# ============================================================================
# TESTS
# ============================================================================


@pytest.mark.parametrize(
    ("fname", "rcut", "box", "cn_mean", "cn_std"),
    [
        ("liquid.xyz", 1.56, np.full(3, 8.54988), 12.21916417, 0.10655101),
        ("solid.xyz", 1.29, np.full(3, 7.46901), 12.00690547, 0.00765176),
    ],
)
def test_cn_calculate(fname, rcut, box, cn_mean, cn_std):
    """Test the CN calculation in LJ liquid and solid."""
    frames = read_xyz(TEST_DATA_PATH / fname)
    result = exma.cn.CoordinationNumber(frames, rcut).calculate(box)

    np.testing.assert_almost_equal(result[0], cn_mean)
    np.testing.assert_almost_equal(result[1], cn_std)


def test_cn_to_dataframe():
    """Test the CN to_dataframe."""
    frames = read_xyz(TEST_DATA_PATH / "solid.xyz")
    cn = exma.cn.CoordinationNumber(frames, 1.29, stop=5)
    cn.calculate(np.full(3, 7.46901))
    df = cn.to_dataframe()

    df_ref = pd.read_csv(str(TEST_DATA_PATH / "cn.csv"))

    pd.testing.assert_frame_equal(df, df_ref)
