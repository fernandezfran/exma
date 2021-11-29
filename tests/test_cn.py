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

import exma.cn

import numpy as np

import pytest

# ============================================================================
# CONSTANTS
# ============================================================================

TEST_DATA_PATH = pathlib.Path(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_data")
)

# ======================================================================
# TESTS
# ======================================================================


@pytest.mark.parametrize(
    ("fname", "rcut", "box", "cn_mean", "cn_std"),
    [
        ("liquid.xyz", 1.56, np.full(3, 8.54988), 12.21916417, 0.10655101),
        ("solid.xyz", 1.29, np.full(3, 7.46901), 12.00690547, 0.00765176),
    ],
)
def test_CoordinationNumber_calculate(fname, rcut, box, cn_mean, cn_std):
    """Test the CN calculation in LJ liquid and solid."""
    result = exma.cn.CoordinationNumber(
        str(TEST_DATA_PATH / fname), "Ar", "Ar", rcut
    ).calculate(box)

    np.testing.assert_almost_equal(result[0], cn_mean)
    np.testing.assert_almost_equal(result[1], cn_std)


@pytest.mark.parametrize(
    "fname",
    ["liquid.out", "solid.xtc", "gas.lammsptrj", "dump.asd.123.lammptsrj"],
)
def test_CoordinationNumber_raises(fname):
    """Test the CN ValueError raise."""
    with pytest.raises(ValueError):
        exma.cn.CoordinationNumber(fname, "H", "H", 1.0).calculate()


@pytest.mark.parametrize(
    ("fname", "rcut", "box"),
    [
        ("liquid.xyz", 1.56, np.full(3, 8.54988)),
        ("solid.xyz", 1.29, np.full(3, 7.46901)),
    ],
)
def test_CoordinationNumber_warning(fname, rcut, box):
    """Test the CN EOF warning."""
    with pytest.warns(UserWarning):
        exma.cn.CoordinationNumber(
            str(TEST_DATA_PATH / fname),
            "Ar",
            "Ar",
            rcut,
            start=190,
            stop=210,
            step=5,
        ).calculate(box)


def test_CoordinationNumber_plot():
    """Test the CN plot."""
    with pytest.raises(NotImplementedError):
        exma.cn.CoordinationNumber("something.xyz", "H", "H", 1.0).plot()


def test_CoordinationNumber_save():
    """Test the CN save."""
    with pytest.raises(NotImplementedError):
        exma.cn.CoordinationNumber("something.xyz", "H", "H", 1.0).save()
