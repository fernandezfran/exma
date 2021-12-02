#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ============================================================================
# IMPORTS
# ============================================================================

import exma.electrochemistry

import numpy as np

import pandas as pd

import pytest

# ============================================================================
# TESTS
# ============================================================================


def test_formation_energy():
    """Test the formation energy."""
    reffe = np.array([0.875, 1.021875, 1.15625, 1.296875, 1.4375])

    x = np.linspace(0, 1, num=5)
    natoms_a = np.full(5, 8)
    epot = np.array([-1.0, -0.825, -0.75, -0.625, -0.5])
    df = pd.DataFrame({"x": x, "natoms_a": natoms_a, "epot": epot})

    result = exma.electrochemistry.formation_energy(df, -1.0, -0.5)

    np.testing.assert_almost_equal(result.x, x)
    np.testing.assert_almost_equal(result.fe, reffe)


def test_formation_energy_error():
    """Test the formation energy error propagation."""
    reffe = np.array([0.875, 1.021875, 1.15625, 1.296875, 1.4375])

    x = np.linspace(0, 1, num=5)
    natoms_a = np.full(5, 8)
    epot = np.array([-1.0, -0.825, -0.75, -0.625, -0.5])
    epot_error = np.full(5, 0.05)
    df = pd.DataFrame(
        {
            "x": x,
            "natoms_a": natoms_a,
            "epot": epot,
            "err_epot": epot_error,
        }
    )

    result = exma.electrochemistry.formation_energy(df, -1.0, -0.5)

    np.testing.assert_almost_equal(result.x, x)
    np.testing.assert_almost_equal(result.fe, reffe)
    np.testing.assert_almost_equal(result.errfe, epot_error)


def test_raise_fe():
    """Test the raise of KeyError in fe."""
    x = np.linspace(0, 1, num=5)
    natoms_a = np.full(5, 8)
    df = pd.DataFrame({"x": x, "natoms_a": natoms_a})

    with pytest.raises(KeyError):
        exma.electrochemistry.formation_energy(df, -1.0, -0.5)
