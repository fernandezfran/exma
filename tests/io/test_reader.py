#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ============================================================================
# IMPORTS
# ============================================================================

import os
import pathlib

import exma.io.reader

import numpy as np

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
    ("traj_dict", "fname", "ftype"),
    [
        (
            {
                "natoms": 5,
                "type": np.array(5 * ["H"]),
                "x": np.array([2.67583, 0.93241, 1.23424, 4.42636, 3.00023]),
                "y": np.array([0.05432, 0.89325, 0.43142, 0.23451, 0.55556]),
                "z": np.array([1.15145, 2.31451, 3.96893, 4.96905, 5.98693]),
            },
            "test_ref.xyz",
            "xyz",
        ),
        (
            {
                "natoms": 5,
                "type": np.array(5 * ["H"]),
                "x": np.array([2.67583, 0.93241, 1.23424, 4.42636, 3.00023]),
                "y": np.array([0.05432, 0.89325, 0.43142, 0.23451, 0.55556]),
                "z": np.array([1.15145, 2.31451, 3.96893, 4.96905, 5.98693]),
                "property": np.arange(0, 5),
            },
            "test_ref.xyz",
            "property",
        ),
        (
            {
                "natoms": 5,
                "type": np.array(5 * ["H"]),
                "x": np.array([2.67583, 0.93241, 1.23424, 4.42636, 3.00023]),
                "y": np.array([0.05432, 0.89325, 0.43142, 0.23451, 0.55556]),
                "z": np.array([1.15145, 2.31451, 3.96893, 4.96905, 5.98693]),
                "ix": np.array([0, 1, 2, 3, 4]),
                "iy": np.array([2, 3, 0, 0, 1]),
                "iz": np.array([-2, -1, 0, 0, 1]),
            },
            "test_ref.xyz",
            "image",
        ),
    ],
)
def test_XYZ(traj_dict, fname, ftype):
    """Test the write of an xyz file."""
    fxyz = TEST_DATA_PATH / fname

    rxyz = exma.io.reader.XYZ(fxyz, ftype)
    result = rxyz.read_frame()
    rxyz.file_close()

    assert result["natoms"] == traj_dict["natoms"]
    np.testing.assert_array_equal(result["type"], traj_dict["type"])
    np.testing.assert_array_almost_equal(result["x"], traj_dict["x"])
    np.testing.assert_array_almost_equal(result["y"], traj_dict["y"])
    np.testing.assert_array_almost_equal(result["z"], traj_dict["z"])

    if "property" in traj_dict.keys():
        np.testing.assert_array_almost_equal(
            result["property"], traj_dict["property"]
        )
    if "ix" in traj_dict.keys():
        np.testing.assert_array_almost_equal(result["ix"], traj_dict["ix"])
        np.testing.assert_array_almost_equal(result["iy"], traj_dict["iy"])
        np.testing.assert_array_almost_equal(result["iz"], traj_dict["iz"])


def test_XYZ_raise_ValueError():
    """Test the ValueError raise read xyz file."""
    fxyz = TEST_DATA_PATH / "test_ref.xyz"

    with pytest.raises(ValueError):
        exma.io.reader.XYZ(fxyz, "error")


def test_XYZ_raise_EOFError():
    """Test the EOFError raise of read xyz file."""
    fxyz = TEST_DATA_PATH / "test_ref.xyz"

    rxyz = exma.io.reader.XYZ(fxyz)
    rxyz.read_frame()
    with pytest.raises(EOFError):
        rxyz.read_frame()
    rxyz.file_close()


@pytest.mark.parametrize(
    ("fname", "frame_dict"),
    [
        (
            "exma_ref.lammpstrj",
            {
                "natoms": 5,
                "box": np.array([4.5, 1.0, 6.0]),
                "id": np.arange(1, 6),
                "type": np.array([1, 1, 1, 2, 2]),
                "x": np.array([2.67583, 0.93241, 1.23424, 4.42636, 3.00023]),
                "y": np.array([0.05432, 0.89325, 0.43142, 0.23451, 0.55556]),
                "z": np.array([1.15145, 2.31451, 3.96893, 4.96905, 5.98693]),
            },
        ),
        (
            "exma_ref.lammpstrj",
            {
                "natoms": 5,
                "box": np.array([4.5, 1.0, 6.0]),
                "id": np.arange(1, 6),
                "type": np.array([1, 1, 1, 2, 2]),
                "q": np.array([-0.3356, -0.32636, -0.34256, 0.54365, 0.46463]),
                "x": np.array([2.67583, 0.93241, 1.23424, 4.42636, 3.00023]),
                "y": np.array([0.05432, 0.89325, 0.43142, 0.23451, 0.55556]),
                "z": np.array([1.15145, 2.31451, 3.96893, 4.96905, 5.98693]),
            },
        ),
        (
            "exma_ref.lammpstrj",
            {
                "natoms": 5,
                "box": np.array([4.5, 1.0, 6.0]),
                "id": np.arange(1, 6),
                "type": np.array([1, 1, 1, 2, 2]),
                "x": np.array([2.67583, 0.93241, 1.23424, 4.42636, 3.00023]),
                "y": np.array([0.05432, 0.89325, 0.43142, 0.23451, 0.55556]),
                "z": np.array([1.15145, 2.31451, 3.96893, 4.96905, 5.98693]),
                "ix": np.array([0, 0, 1, -1, 0]),
                "iy": np.array([2, 3, 0, 0, 1]),
                "iz": np.array([-2, -1, 0, 0, 1]),
            },
        ),
        (
            "exma_ref.lammpstrj",
            {
                "natoms": 5,
                "box": np.array([4.5, 1.0, 6.0]),
                "id": np.arange(1, 6),
                "type": np.array([1, 1, 1, 2, 2]),
                "q": np.array([-0.3356, -0.32636, -0.34256, 0.54365, 0.46463]),
                "x": np.array([2.67583, 0.93241, 1.23424, 4.42636, 3.00023]),
                "y": np.array([0.05432, 0.89325, 0.43142, 0.23451, 0.55556]),
                "z": np.array([1.15145, 2.31451, 3.96893, 4.96905, 5.98693]),
                "ix": np.array([0, 0, 1, -1, 0]),
                "iy": np.array([2, 3, 0, 0, 1]),
                "iz": np.array([-2, -1, 0, 0, 1]),
            },
        ),
    ],
)
def test_LAMMPS(fname, frame_dict):
    """Test the read of an .lammpstrj file."""
    flmp = TEST_DATA_PATH / fname

    rlmp = exma.io.reader.LAMMPS(flmp)
    result = rlmp.read_frame()
    rlmp.file_close()

    for key in frame_dict.keys():
        np.testing.assert_array_almost_equal(result[key], frame_dict[key])


def test_LAMMPS_raises():
    """Test the ValueError raise of write .lammpstrj file."""
    flmp = TEST_DATA_PATH / "exma_ref.lammpstrj"
    rlmp = exma.io.reader.LAMMPS(flmp)
    rlmp.read_frame()
    with pytest.raises(EOFError):
        rlmp.read_frame()
    rlmp.file_close()


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
