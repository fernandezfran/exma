#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ======================================================================
# IMPORTS
# ======================================================================

import exma.io.positions

import numpy as np

import pytest

# ======================================================================
# TESTS
# ======================================================================


def test_sc():
    """Test that the atoms are placed in a simple cubic crystal."""
    boxref = np.full(3, 1.0)
    xref = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])
    yref = np.array([0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5])
    zref = np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5])

    result = exma.io.positions.Positions(8, 1.0).sc()

    assert result["natoms"] == 8
    np.testing.assert_array_equal(result["box"], boxref)
    np.testing.assert_array_equal(result["x"], xref)
    np.testing.assert_array_equal(result["y"], yref)
    np.testing.assert_array_equal(result["z"], zref)


def test_bcc():
    """Test that the atoms are placed in a body-centered cubic crystal."""
    boxref = np.full(3, 1.0)
    xref = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.5,
            0.5,
            0.5,
            0.5,
            0.25,
            0.25,
            0.25,
            0.25,
            0.75,
            0.75,
            0.75,
            0.75,
        ]
    )
    yref = np.array(
        [
            0.0,
            0.0,
            0.5,
            0.5,
            0.0,
            0.0,
            0.5,
            0.5,
            0.25,
            0.25,
            0.75,
            0.75,
            0.25,
            0.25,
            0.75,
            0.75,
        ]
    )
    zref = np.array(
        [
            0.0,
            0.5,
            0.0,
            0.5,
            0.0,
            0.5,
            0.0,
            0.5,
            0.25,
            0.75,
            0.25,
            0.75,
            0.25,
            0.75,
            0.25,
            0.75,
        ]
    )

    result = exma.io.positions.Positions(16, 1.0).bcc()

    assert result["natoms"] == 16
    np.testing.assert_array_equal(result["box"], boxref)
    np.testing.assert_array_equal(result["x"], xref)
    np.testing.assert_array_equal(result["y"], yref)
    np.testing.assert_array_equal(result["z"], zref)


def test_fcc():
    """Test that the atoms are placed in a face-centered cubic crystal."""
    boxref = np.full(3, 1.0)
    xref = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.5,
            0.5,
            0.5,
            0.5,
            0.25,
            0.25,
            0.25,
            0.25,
            0.75,
            0.75,
            0.75,
            0.75,
            0.25,
            0.25,
            0.25,
            0.25,
            0.75,
            0.75,
            0.75,
            0.75,
            0.0,
            0.0,
            0.0,
            0.0,
            0.5,
            0.5,
            0.5,
            0.5,
        ]
    )
    yref = np.array(
        [
            0.0,
            0.0,
            0.5,
            0.5,
            0.0,
            0.0,
            0.5,
            0.5,
            0.25,
            0.25,
            0.75,
            0.75,
            0.25,
            0.25,
            0.75,
            0.75,
            0.0,
            0.0,
            0.5,
            0.5,
            0.0,
            0.0,
            0.5,
            0.5,
            0.25,
            0.25,
            0.75,
            0.75,
            0.25,
            0.25,
            0.75,
            0.75,
        ]
    )
    zref = np.array(
        [
            0.0,
            0.5,
            0.0,
            0.5,
            0.0,
            0.5,
            0.0,
            0.5,
            0.0,
            0.5,
            0.0,
            0.5,
            0.0,
            0.5,
            0.0,
            0.5,
            0.25,
            0.75,
            0.25,
            0.75,
            0.25,
            0.75,
            0.25,
            0.75,
            0.25,
            0.75,
            0.25,
            0.75,
            0.25,
            0.75,
            0.25,
            0.75,
        ]
    )

    result = exma.io.positions.Positions(32, 1.0).fcc()

    assert result["natoms"] == 32
    np.testing.assert_array_equal(result["box"], boxref)
    np.testing.assert_array_equal(result["x"], xref)
    np.testing.assert_array_equal(result["y"], yref)
    np.testing.assert_array_equal(result["z"], zref)


def test_dc():
    """Test that the atoms are placed in a diamond cubic crystal."""
    boxref = np.full(3, 1.0)
    xref = np.array([0.25, 0.0, 0.25, 0.0, 0.75, 0.5, 0.75, 0.5])
    yref = np.array([0.75, 0.0, 0.25, 0.5, 0.75, 0.0, 0.25, 0.5])
    zref = np.array([0.25, 0.5, 0.75, 0.0, 0.75, 0.0, 0.25, 0.5])

    result = exma.io.positions.Positions(8, 1.0).dc()

    assert result["natoms"] == 8
    np.testing.assert_array_equal(result["box"], boxref)
    np.testing.assert_array_equal(result["x"], xref)
    np.testing.assert_array_equal(result["y"], yref)
    np.testing.assert_array_equal(result["z"], zref)


def test_sc_raise():
    """Test the raise of sc."""
    particles = exma.io.positions.Positions(7, 1.0)
    with pytest.raises(ValueError):
        particles.sc()


def test_bcc_raise():
    """Test the raise of the bcc."""
    particles = exma.io.positions.Positions(19, 1.0)
    with pytest.raises(ValueError):
        particles.bcc()


def test_fcc_raise():
    """Test the raise of the fcc."""
    particles = exma.io.positions.Positions(37, 1.0)
    with pytest.raises(ValueError):
        particles.fcc()


def test_dc_raise():
    """Test the raise of the dc."""
    particles = exma.io.positions.Positions(9, 1.0)
    with pytest.raises(ValueError):
        particles.dc()


def test_spherical_nanoparticle():
    """Test the spherical nanoparticle"""
    xref = np.array([-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    yref = np.array([0.0, -0.5, 0.0, 0.0, 0.0, 0.5, 0.0])
    zref = np.array([0.0, 0.0, -0.5, 0.0, 0.5, 0.0, 0.0])

    frame = {
        "box": np.full(3, 1.0),
        "x": np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5]),
        "y": np.array([0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5]),
        "z": np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5]),
    }

    result = exma.io.positions.spherical_nanoparticle(frame, 0.6)

    assert result["natoms"] == 7
    np.testing.assert_array_equal(result["x"], xref)
    np.testing.assert_array_equal(result["y"], yref)
    np.testing.assert_array_equal(result["z"], zref)


def test_replicate():
    """Test the replicate function."""
    natomsref = 8 * 2 * 2 * 2
    boxref = np.full(3, 2 * 5.468728)
    typesref = ["Si"] * 8 * 2 * 2 * 2
    xref = np.array(
        [
            1.367182,
            0.000000,
            1.367182,
            0.000000,
            4.101546,
            2.734364,
            4.101546,
            2.734364,
            1.367182,
            0.000000,
            1.367182,
            0.000000,
            4.101546,
            2.734364,
            4.101546,
            2.734364,
            1.367182,
            0.000000,
            1.367182,
            0.000000,
            4.101546,
            2.734364,
            4.101546,
            2.734364,
            1.367182,
            0.000000,
            1.367182,
            0.000000,
            4.101546,
            2.734364,
            4.101546,
            2.734364,
            6.83591,
            5.468728,
            6.83591,
            5.468728,
            9.570274,
            8.203092,
            9.570274,
            8.203092,
            6.83591,
            5.468728,
            6.83591,
            5.468728,
            9.570274,
            8.203092,
            9.570274,
            8.203092,
            6.83591,
            5.468728,
            6.83591,
            5.468728,
            9.570274,
            8.203092,
            9.570274,
            8.203092,
            6.83591,
            5.468728,
            6.83591,
            5.468728,
            9.570274,
            8.203092,
            9.570274,
            8.203092,
        ]
    )
    yref = np.array(
        [
            4.101546,
            0.000000,
            1.367182,
            2.734364,
            4.101546,
            0.000000,
            1.367182,
            2.734364,
            4.101546,
            0.000000,
            1.367182,
            2.734364,
            4.101546,
            0.000000,
            1.367182,
            2.734364,
            9.570274,
            5.468728,
            6.83591,
            8.203092,
            9.570274,
            5.468728,
            6.83591,
            8.203092,
            9.570274,
            5.468728,
            6.83591,
            8.203092,
            9.570274,
            5.468728,
            6.83591,
            8.203092,
            4.101546,
            0.000000,
            1.367182,
            2.734364,
            4.101546,
            0.000000,
            1.367182,
            2.734364,
            4.101546,
            0.000000,
            1.367182,
            2.734364,
            4.101546,
            0.000000,
            1.367182,
            2.734364,
            9.570274,
            5.468728,
            6.83591,
            8.203092,
            9.570274,
            5.468728,
            6.83591,
            8.203092,
            9.570274,
            5.468728,
            6.83591,
            8.203092,
            9.570274,
            5.468728,
            6.83591,
            8.203092,
        ]
    )
    zref = np.array(
        [
            1.367182,
            2.734364,
            4.101546,
            0.000000,
            4.101546,
            0.000000,
            1.367182,
            2.734364,
            6.83591,
            8.203092,
            9.570274,
            5.468728,
            9.570274,
            5.468728,
            6.83591,
            8.203092,
            1.367182,
            2.734364,
            4.101546,
            0.000000,
            4.101546,
            0.000000,
            1.367182,
            2.734364,
            6.83591,
            8.203092,
            9.570274,
            5.468728,
            9.570274,
            5.468728,
            6.83591,
            8.203092,
            1.367182,
            2.734364,
            4.101546,
            0.000000,
            4.101546,
            0.000000,
            1.367182,
            2.734364,
            6.83591,
            8.203092,
            9.570274,
            5.468728,
            9.570274,
            5.468728,
            6.83591,
            8.203092,
            1.367182,
            2.734364,
            4.101546,
            0.000000,
            4.101546,
            0.000000,
            1.367182,
            2.734364,
            6.83591,
            8.203092,
            9.570274,
            5.468728,
            9.570274,
            5.468728,
            6.83591,
            8.203092,
        ]
    )

    frame = {
        "natoms": 8,
        "box": np.full(3, 5.468728),
        "type": ["Si"] * 8,
        "x": np.array([0.25, 0.0, 0.25, 0.0, 0.75, 0.5, 0.75, 0.5]),
        "y": np.array([0.75, 0.0, 0.25, 0.5, 0.75, 0.0, 0.25, 0.5]),
        "z": np.array([0.25, 0.5, 0.75, 0.0, 0.75, 0.0, 0.25, 0.5]),
    }

    result = exma.io.positions.replicate(frame, [2, 2, 2])

    assert result["natoms"] == natomsref
    np.testing.assert_array_almost_equal(result["box"], boxref)
    np.testing.assert_array_equal(result["type"], typesref)
    np.testing.assert_array_almost_equal(result["x"], xref)
    np.testing.assert_array_almost_equal(result["y"], yref)
    np.testing.assert_array_almost_equal(result["z"], zref)
