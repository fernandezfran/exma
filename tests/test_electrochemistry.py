#!/usr/bin/env python
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

import exma.electrochemistry

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
    np.testing.assert_almost_equal(result.err_fvc, referrfvc)


def test_raise_fvc():
    """Test the raise of KeyError in fvc."""
    x = np.linspace(0, 1, num=5)
    natoms_a = np.full(5, 8)
    df = pd.DataFrame({"x": x, "natoms_a": natoms_a})

    with pytest.raises(KeyError):
        exma.electrochemistry.fractional_volume_change(df, 8, 125)


def test_formation_energy_x():
    """Test the formation energy."""
    reffe = np.array([0.875, 1.021875, 1.15625, 1.296875, 1.4375])

    x = np.linspace(0, 1, num=5)
    natoms_a = np.full(5, 8)
    epot = np.array([-1.0, -0.825, -0.75, -0.625, -0.5])
    df = pd.DataFrame({"x": x, "natoms_a": natoms_a, "epot": epot})

    result = exma.electrochemistry.formation_energy(df, -1.0, -0.5)

    np.testing.assert_almost_equal(result.x, x)
    np.testing.assert_almost_equal(result.fe, reffe)


def test_formation_energy_relative():
    """Test the formation energy."""
    df = pd.read_csv(TEST_DATA_PATH / "cristals.csv")
    reffe = np.array(
        [0.0, -0.21365, -0.209494, -0.229424, -0.172029, -0.302786, 0.0]
    )

    result = exma.electrochemistry.formation_energy(
        df, -1.62041487, -4.71777201, fetype="relative"
    )

    np.testing.assert_almost_equal(result.fe, reffe, decimal=5)


def test_formation_energy_x_error():
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
    np.testing.assert_almost_equal(result.err_fe, epot_error / natoms_a)


def test_formation_energy_relative_error():
    """Test the formation energy error propagation."""
    df = pd.read_csv(TEST_DATA_PATH / "cristals.csv")
    epot_error = np.full(7, 0.1)
    df["err_epot"] = epot_error
    reffe = np.array(
        [0.0, -0.21365, -0.209494, -0.229424, -0.172029, -0.302786, 0.0]
    )

    result = exma.electrochemistry.formation_energy(
        df, -1.62041487, -4.71777201, fetype="relative"
    )

    np.testing.assert_almost_equal(result.fe, reffe, decimal=5)
    print(result.err_fe, epot_error / (df.natoms_a + df.natoms_b))
    np.testing.assert_almost_equal(
        result.err_fe.to_numpy(),
        (epot_error / (df.natoms_a + df.natoms_b)).to_numpy(),
    )


def test_raise_fe():
    """Test the raise of KeyError in fe."""
    x = np.linspace(0, 1, num=5)
    natoms_a = np.full(5, 8)
    df = pd.DataFrame({"x": x, "natoms_a": natoms_a})

    with pytest.raises(KeyError):
        exma.electrochemistry.formation_energy(df, -1.0, -0.5)


def test_voltage():
    """Test the voltage approximation."""
    reffe_spline = np.array(
        [
            0.8766071,
            0.8881785,
            0.8997439,
            0.9113034,
            0.9228569,
            0.9344045,
            0.9459461,
            0.9574818,
            0.9690115,
            0.98053527,
            0.9920531,
            1.003565,
            1.0150709,
            1.0265709,
            1.0380648,
            1.0495529,
            1.061035,
            1.0725112,
            1.0839815,
            1.0954458,
            1.106904,
            1.1183565,
            1.1298028,
            1.1412433,
            1.1526779,
            1.1641064,
            1.175529,
            1.1869457,
            1.1983564,
            1.2097611,
            1.22116,
            1.2325529,
            1.2439398,
            1.2553208,
            1.2666959,
            1.2780648,
            1.289428,
            1.3007852,
            1.3121364,
            1.3234817,
            1.334821,
            1.3461543,
            1.3574818,
            1.3688033,
            1.3801187,
            1.3914284,
            1.402732,
            1.4140296,
            1.4253213,
            1.4366071,
        ],
    )
    refvoltage = np.array(
        [
            -0.5671429,
            -0.5668513,
            -0.5665598,
            -0.56626827,
            -0.5659767,
            -0.56568515,
            -0.5653936,
            -0.56510204,
            -0.5648105,
            -0.5645189,
            -0.5642274,
            -0.5639359,
            -0.5636443,
            -0.56335276,
            -0.56306124,
            -0.56276965,
            -0.5624781,
            -0.5621866,
            -0.561895,
            -0.5616035,
            -0.56131196,
            -0.5610204,
            -0.56072885,
            -0.5604373,
            -0.56014574,
            -0.5598542,
            -0.5595627,
            -0.5592711,
            -0.5589796,
            -0.55868804,
            -0.55839646,
            -0.55810493,
            -0.5578134,
            -0.5575218,
            -0.5572303,
            -0.55693877,
            -0.5566472,
            -0.55635566,
            -0.5560641,
            -0.55577254,
            -0.555481,
            -0.5551895,
            -0.5548979,
            -0.5546064,
            -0.55431485,
            -0.55402327,
            -0.55373174,
            -0.5534402,
            -0.5531486,
            -0.5528571,
        ],
    )

    x = np.linspace(0, 1, num=5)
    fe = np.array([0.875000, 1.021875, 1.156250, 1.296875, 1.437500])
    df = pd.DataFrame({"x": x, "fe": fe})

    result = exma.electrochemistry.voltage(df, k=2)

    np.testing.assert_almost_equal(result.x, np.linspace(0, 1))
    np.testing.assert_almost_equal(result.fe_spline, reffe_spline, 6)
    np.testing.assert_almost_equal(result.voltage, refvoltage)
