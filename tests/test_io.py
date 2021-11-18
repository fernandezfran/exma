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

import exma.reader
import exma.writer

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


def test_xyz():
    """Test the write and read of an xyz file."""
    natoms = 5
    types = ["H"] * natoms
    xyz = np.array(
        [
            2.67583,
            0.93241,
            1.23424,
            4.42636,
            3.00023,
            0.05432,
            0.89325,
            0.43142,
            0.23451,
            0.55556,
            1.15145,
            2.31451,
            3.96893,
            4.96905,
            5.98693,
        ]
    )

    wxyz = exma.writer.xyz(TEST_DATA_PATH / "test.xyz")
    wxyz.write_frame(natoms, types, xyz)
    wxyz.file_close()

    rxyz = exma.reader.xyz(TEST_DATA_PATH / "test.xyz")
    result = rxyz.read_frame()
    rxyz.file_close()

    assert result[0] == natoms
    assert result[1] == types
    np.testing.assert_array_almost_equal(result[2], xyz)


def test_xyz_property():
    """Test the write and read xyz file with a property in the last column."""
    natoms = 5
    types = ["H"] * natoms
    xyz = np.array(
        [
            2.67583,
            0.93241,
            1.23424,
            4.42636,
            3.00023,
            0.05432,
            0.89325,
            0.43142,
            0.23451,
            0.55556,
            1.15145,
            2.31451,
            3.96893,
            4.96905,
            5.98693,
        ]
    )
    prop = np.arange(0, 5)

    wxyz = exma.writer.xyz(TEST_DATA_PATH / "test.xyz", "property")
    wxyz.write_frame(natoms, types, xyz, prop)
    wxyz.file_close()

    rxyz = exma.reader.xyz(TEST_DATA_PATH / "test.xyz", "property")
    result = rxyz.read_frame()
    rxyz.file_close()

    assert result[0] == natoms
    assert result[1] == types
    np.testing.assert_array_almost_equal(result[2], xyz)
    np.testing.assert_array_almost_equal(result[3], prop)


def test_xyz_image():
    """Test the write and read xyz file with images in the last columns."""
    natoms = 5
    types = ["H"] * natoms
    xyz = np.array(
        [
            2.67583,
            0.93241,
            1.23424,
            4.42636,
            3.00023,
            0.05432,
            0.89325,
            0.43142,
            0.23451,
            0.55556,
            1.15145,
            2.31451,
            3.96893,
            4.96905,
            5.98693,
        ]
    )
    img = np.array([0, 0, 1, -1, 0, 2, 3, 0, 0, 1, -2, -1, 0, 0, 1])

    wxyz = exma.writer.xyz(TEST_DATA_PATH / "test.xyz", "image")
    wxyz.write_frame(natoms, types, xyz, image=img)
    wxyz.file_close()

    rxyz = exma.reader.xyz(TEST_DATA_PATH / "test.xyz", "image")
    result = rxyz.read_frame()
    rxyz.file_close()

    assert result[0] == natoms
    assert result[1] == types
    np.testing.assert_array_almost_equal(result[2], xyz)
    np.testing.assert_array_almost_equal(result[3], img)


def test_xyz_raises():
    """Test the raises of write and read xyz file."""
    with pytest.raises(ValueError):
        exma.writer.xyz(TEST_DATA_PATH / "test.xyz", "error")

    with pytest.raises(ValueError):
        exma.reader.xyz(TEST_DATA_PATH / "test.xyz", "error")

    rxyz = exma.reader.xyz(TEST_DATA_PATH / "test.xyz")
    rxyz.read_frame()
    with pytest.raises(EOFError):
        rxyz.read_frame()
    rxyz.file_close()


def test_lammpstrj():
    """Test the write and read .lammpstrj file."""
    natoms = 5
    box = np.array([4.5, 1.0, 6.0])
    idx = np.arange(1, natoms + 1)
    types = np.array([1, 1, 1, 2, 2])
    xyz = np.array(
        [
            2.67583,
            0.93241,
            1.23424,
            4.42636,
            3.00023,
            0.05432,
            0.89325,
            0.43142,
            0.23451,
            0.55556,
            1.15145,
            2.31451,
            3.96893,
            4.96905,
            5.98693,
        ]
    )

    wlmp = exma.writer.lammpstrj(TEST_DATA_PATH / "test.lammpstrj")
    wlmp.write_frame(natoms, box, idx, types, xyz)
    wlmp.file_close()

    rlmp = exma.reader.lammpstrj(TEST_DATA_PATH / "test.lammpstrj")
    result = rlmp.read_frame()
    rlmp.file_close()

    assert result[0] == natoms
    np.testing.assert_array_almost_equal(result[1], box)
    np.testing.assert_array_equal(result[2], idx)
    np.testing.assert_array_equal(result[3], types)
    np.testing.assert_array_almost_equal(result[4], xyz)


def test_lammpstrj_charge():
    """Test the write and read .lammpstrj file with charges."""
    natoms = 5
    box = np.array([4.5, 1.0, 6.0])
    idx = np.arange(1, natoms + 1)
    types = np.array([1, 1, 1, 2, 2])
    xyz = np.array(
        [
            2.67583,
            0.93241,
            1.23424,
            4.42636,
            3.00023,
            0.05432,
            0.89325,
            0.43142,
            0.23451,
            0.55556,
            1.15145,
            2.31451,
            3.96893,
            4.96905,
            5.98693,
        ]
    )
    charges = np.array([-0.3356, -0.32636, -0.34256, 0.54365, 0.46463])

    wlmp = exma.writer.lammpstrj(TEST_DATA_PATH / "test.lammpstrj", "charge")
    wlmp.write_frame(natoms, box, idx, types, xyz, charges)
    wlmp.file_close()

    rlmp = exma.reader.lammpstrj(TEST_DATA_PATH / "test.lammpstrj", "charge")
    result = rlmp.read_frame()
    rlmp.file_close()

    assert result[0] == natoms
    np.testing.assert_array_almost_equal(result[1], box)
    np.testing.assert_array_equal(result[2], idx)
    np.testing.assert_array_equal(result[3], types)
    np.testing.assert_array_almost_equal(result[4], xyz)
    np.testing.assert_array_almost_equal(result[5], charges)


def test_lammpstrj_image():
    """Test write and read .lammpstrj file with images box."""
    natoms = 5
    box = np.array([4.5, 1.0, 6.0])
    idx = np.arange(1, natoms + 1)
    types = np.array([1, 1, 1, 2, 2])
    xyz = np.array(
        [
            2.67583,
            0.93241,
            1.23424,
            4.42636,
            3.00023,
            0.05432,
            0.89325,
            0.43142,
            0.23451,
            0.55556,
            1.15145,
            2.31451,
            3.96893,
            4.96905,
            5.98693,
        ]
    )
    img = np.array([0, 0, 1, -1, 0, 2, 3, 0, 0, 1, -2, -1, 0, 0, 1])

    wlmp = exma.writer.lammpstrj(TEST_DATA_PATH / "test.lammpstrj", "image")
    wlmp.write_frame(natoms, box, idx, types, xyz, image=img)
    wlmp.file_close()

    rlmp = exma.reader.lammpstrj(TEST_DATA_PATH / "test.lammpstrj", "image")
    result = rlmp.read_frame()
    rlmp.file_close()

    assert result[0] == natoms
    np.testing.assert_array_almost_equal(result[1], box)
    np.testing.assert_array_equal(result[2], idx)
    np.testing.assert_array_equal(result[3], types)
    np.testing.assert_array_almost_equal(result[4], xyz)
    np.testing.assert_array_equal(result[5], img)


def test_lammpstrj_charge_image():
    """Test the write and read .lammpstrj file with images box and charges."""
    natoms = 5
    box = np.array([4.5, 1.0, 6.0])
    idx = np.arange(1, natoms + 1)
    types = np.array([1, 1, 1, 2, 2])
    xyz = np.array(
        [
            2.67583,
            0.93241,
            1.23424,
            4.42636,
            3.00023,
            0.05432,
            0.89325,
            0.43142,
            0.23451,
            0.55556,
            1.15145,
            2.31451,
            3.96893,
            4.96905,
            5.98693,
        ]
    )
    charges = np.array([-0.3356, -0.32636, -0.34256, 0.54365, 0.46463])
    img = np.array([0, 0, 1, -1, 0, 2, 3, 0, 0, 1, -2, -1, 0, 0, 1])

    wlmp = exma.writer.lammpstrj(TEST_DATA_PATH / "test.lammpstrj", "charge_image")
    wlmp.write_frame(natoms, box, idx, types, xyz, charges, img)
    wlmp.file_close()

    rlmp = exma.reader.lammpstrj(TEST_DATA_PATH / "test.lammpstrj", "charge_image")
    result = rlmp.read_frame()
    rlmp.file_close()

    assert result[0] == natoms
    np.testing.assert_array_almost_equal(result[1], box)
    np.testing.assert_array_equal(result[2], idx)
    np.testing.assert_array_equal(result[3], types)
    np.testing.assert_array_almost_equal(result[4], xyz)
    np.testing.assert_array_almost_equal(result[5], charges)
    np.testing.assert_array_equal(result[6], img)

def test_lammpstrj_raises():
    """Test the raises of write and read .lammpstrj file."""
    with pytest.raises(ValueError):
        exma.writer.xyz(TEST_DATA_PATH / "test.lammpstrj", "error")

    with pytest.raises(ValueError):
        exma.reader.xyz(TEST_DATA_PATH / "test.lammpstrj", "error")

    rlmp = exma.reader.lammpstrj(TEST_DATA_PATH / "test.lammpstrj")
    rlmp.read_frame()
    with pytest.raises(EOFError):
        rlmp.read_frame()
    rlmp.file_close()
