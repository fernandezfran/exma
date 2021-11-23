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
    natomsa = np.full(5, 8)
    volume = np.array([150, 175, 200, 225, 250])
    df = pd.DataFrame({"x": x, "NA": natomsa, "Vol": volume})

    result = exma.electrochemistry.Electrochemistry(
        df
    ).fractional_volume_change(8, 125)

    np.testing.assert_almost_equal(result.x, x)
    np.testing.assert_almost_equal(result.fvc, reffvc)


def test_fractional_volume_change_err():
    """Test the fractional volume change with error propagation."""
    reffvc = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    referrfvc = np.full(5, 0.04)

    x = np.linspace(0, 1, num=5)
    natomsa = np.full(5, 8)
    volume = np.array([150, 175, 200, 225, 250])
    volume_error = np.full(5, 5)
    df = pd.DataFrame(
        {"x": x, "NA": natomsa, "Vol": volume, "errVol": volume_error}
    )

    result = exma.electrochemistry.Electrochemistry(
        df
    ).fractional_volume_change(8, 125)

    np.testing.assert_almost_equal(result.x, x)
    np.testing.assert_almost_equal(result.fvc, reffvc)
    np.testing.assert_almost_equal(result.errfvc, referrfvc)


def test_formation_energy():
    """Test the formation energy."""
    reffe = np.array([0.875, 1.021875, 1.15625, 1.296875, 1.4375])

    x = np.linspace(0, 1, num=5)
    natomsa = np.full(5, 8)
    potential_energy = np.array([-1.0, -0.825, -0.75, -0.625, -0.5])
    df = pd.DataFrame({"x": x, "NA": natomsa, "PotEng": potential_energy})

    result = exma.electrochemistry.Electrochemistry(df).formation_energy(
        -1.0, -0.5
    )

    np.testing.assert_almost_equal(result.x, x)
    np.testing.assert_almost_equal(result.fe, reffe)


def test_formation_energy_error():
    """Test the formation energy error propagation."""
    reffe = np.array([0.875, 1.021875, 1.15625, 1.296875, 1.4375])

    x = np.linspace(0, 1, num=5)
    natomsa = np.full(5, 8)
    potential_energy = np.array([-1.0, -0.825, -0.75, -0.625, -0.5])
    potential_energy_error = np.full(5, 0.05)
    df = pd.DataFrame(
        {
            "x": x,
            "NA": natomsa,
            "PotEng": potential_energy,
            "errPotEng": potential_energy_error,
        }
    )

    result = exma.electrochemistry.Electrochemistry(df).formation_energy(
        -1.0, -0.5
    )

    np.testing.assert_almost_equal(result.x, x)
    np.testing.assert_almost_equal(result.fe, reffe)
    np.testing.assert_almost_equal(result.errfe, potential_energy_error)


def test_voltage():
    """Test the voltage approximation."""
    # reffe = np.array([0.875, 1.021875, 1.15625, 1.296875, 1.4375])
    reffe_spline = np.array(
        [0.8766071, 1.0179465, 1.1583929, 1.2979465, 1.4366071]
    )
    refvoltage = np.array(
        [-0.5671429, -0.5635715, -0.56, -0.5564286, -0.5528571]
    )

    x = np.linspace(0, 1, num=5)
    natomsa = np.full(5, 8)
    potential_energy = np.array([-1.0, -0.825, -0.75, -0.625, -0.5])
    df = pd.DataFrame({"x": x, "NA": natomsa, "PotEng": potential_energy})

    electro = exma.electrochemistry.Electrochemistry(df)
    electro.formation_energy(-1.0, -0.5)
    result = electro.voltage(k=2)

    np.testing.assert_almost_equal(result.x, x)
    np.testing.assert_almost_equal(result.fe_spline, reffe_spline)
    np.testing.assert_almost_equal(result.voltage, refvoltage)


def test_raise_fvc():
    """Test the raise of KeyError in fvc."""
    x = np.linspace(0, 1, num=5)
    natomsa = np.full(5, 8)
    df = pd.DataFrame({"x": x, "NA": natomsa})

    with pytest.raises(KeyError):
        exma.electrochemistry.Electrochemistry(df).fractional_volume_change(
            8, 125
        )


def test_raise_fe():
    """Test the raise of KeyError in fe."""
    x = np.linspace(0, 1, num=5)
    natomsa = np.full(5, 8)
    df = pd.DataFrame({"x": x, "NA": natomsa})

    with pytest.raises(KeyError):
        exma.electrochemistry.Electrochemistry(df).formation_energy(-1.0, -0.5)


def test_raise_voltage():
    """Test the raise of KeyError in voltage."""
    x = np.linspace(0, 1, num=5)
    potential_energy = np.array([-1.0, -0.825, -0.75, -0.625, -0.5])
    df = pd.DataFrame({"x": x, "PotEng": potential_energy})

    electro = exma.electrochemistry.Electrochemistry(df)
    with pytest.raises(NameError):
        electro.voltage(k=2)
