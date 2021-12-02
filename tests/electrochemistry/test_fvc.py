#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ======================================================================
# IMPORTS
# ======================================================================

import exma.electrochemistry

import numpy as np

import pandas as pd

import pytest

# ======================================================================
# TESTS
# ======================================================================


def test_fractional_volume_change():
    """Test the fractional volume change."""
    reffvc = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

    x = np.linspace(0, 1, num=5)
    natoms_a = np.full(5, 8)
    volume = np.array([150, 175, 200, 225, 250])
    df = pd.DataFrame({"x": x, "natoms_a": natoms_a, "volume": volume})

    result = exma.electrochemistry.fractional_volume_change(df, 8, 125)

    np.testing.assert_almost_equal(result.x, x)
    np.testing.assert_almost_equal(result.fvc, reffvc)


def test_fractional_volume_change_err():
    """Test the fractional volume change with error propagation."""
    reffvc = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    referrfvc = np.full(5, 0.04)

    x = np.linspace(0, 1, num=5)
    natoms_a = np.full(5, 8)
    volume = np.array([150, 175, 200, 225, 250])
    volume_error = np.full(5, 5)
    df = pd.DataFrame(
        {
            "x": x,
            "natoms_a": natoms_a,
            "volume": volume,
            "err_volume": volume_error,
        }
    )

    result = exma.electrochemistry.fractional_volume_change(df, 8, 125)

    np.testing.assert_almost_equal(result.x, x)
    np.testing.assert_almost_equal(result.fvc, reffvc)
    np.testing.assert_almost_equal(result.errfvc, referrfvc)


def test_raise_fvc():
    """Test the raise of KeyError in fvc."""
    x = np.linspace(0, 1, num=5)
    natoms_a = np.full(5, 8)
    df = pd.DataFrame({"x": x, "natoms_a": natoms_a})

    with pytest.raises(KeyError):
        exma.electrochemistry.fractional_volume_change(df, 8, 125)
