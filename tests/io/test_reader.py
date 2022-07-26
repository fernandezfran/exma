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


def test_read_xyz():
    """Test the read of xyz file"""
    frames = exma.io.reader.read_xyz(TEST_DATA_PATH / "read_xyz.xyz")

    natoms = 5
    types = np.array(5 * ["H"])
    x = np.array([2.67583, 0.93241, 1.23424, 4.42636, 3.00023])
    y = np.array([0.05432, 0.89325, 0.43142, 0.23451, 0.55556])
    z = np.array([1.15145, 2.31451, 3.96893, 4.96905, 5.98693])

    for k, frame in enumerate(frames):
        assert frame.natoms == natoms
        np.testing.assert_array_equal(frame.types, types)
        np.testing.assert_array_almost_equal(
            frame.x, x * 10**k, decimal=1e-5
        )
        np.testing.assert_array_almost_equal(frame.y, y)
        np.testing.assert_array_almost_equal(frame.z, z)


def test_read_lammpstrj():
    """Test the read of lammpstrj file"""
    frames = exma.io.reader.read_lammpstrj(
        TEST_DATA_PATH / "read_lammpstrj.lammpstrj"
    )

    natoms = 5
    box = np.array([4.5, 1.0, 6.0])
    idx = np.arange(1, 6)
    types = np.array([1, 1, 1, 2, 2])
    x = np.array([2.67583, 0.93241, 1.23424, 4.42636, 3.00023])
    y = np.array([0.05432, 0.89325, 0.43142, 0.23451, 0.55556])
    z = np.array([1.15145, 2.31451, 3.96893, 4.96905, 5.98693])

    for k, frame in enumerate(frames):
        assert frame.natoms == natoms

        np.testing.assert_array_almost_equal(frame.box, box)
        np.testing.assert_array_equal(frame.idx, idx)
        np.testing.assert_array_equal(frame.types, types)

        np.testing.assert_array_almost_equal(frame.x, x)
        np.testing.assert_array_almost_equal(frame.y, y)
        np.testing.assert_array_almost_equal(frame.z, z + k)


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
def test_xyz(traj_dict, fname, ftype):
    """Test the write of an xyz file."""
    fxyz = TEST_DATA_PATH / fname

    with exma.io.reader.XYZ(fxyz, ftype) as rxyz:
        result = rxyz.read_frame()

    assert result.natoms == traj_dict["natoms"]
    np.testing.assert_array_equal(result.types, traj_dict["type"])
    np.testing.assert_array_almost_equal(result.x, traj_dict["x"])
    np.testing.assert_array_almost_equal(result.y, traj_dict["y"])
    np.testing.assert_array_almost_equal(result.z, traj_dict["z"])

    if "property" in traj_dict.keys():
        np.testing.assert_array_almost_equal(result.q, traj_dict["property"])
    if "ix" in traj_dict.keys():
        np.testing.assert_array_almost_equal(result.ix, traj_dict["ix"])
        np.testing.assert_array_almost_equal(result.iy, traj_dict["iy"])
        np.testing.assert_array_almost_equal(result.iz, traj_dict["iz"])


def test_xyz_veraise():
    """Test the ValueError raise read xyz file."""
    fxyz = TEST_DATA_PATH / "test_ref.xyz"

    with pytest.raises(ValueError):
        exma.io.reader.XYZ(fxyz, "error")


def test_xyz_eofraise():
    """Test the EOFError raise of read xyz file."""
    fxyz = TEST_DATA_PATH / "test_ref.xyz"

    with pytest.raises(EOFError):
        with exma.io.reader.XYZ(fxyz) as rxyz:
            rxyz.read_frame()
            rxyz.read_frame()


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
def test_lammps(fname, frame_dict):
    """Test the read of an .lammpstrj file."""
    flmp = TEST_DATA_PATH / fname

    with exma.io.reader.LAMMPS(flmp) as rlmp:
        result = rlmp.read_frame()

    for key in frame_dict.keys():
        if key == "id":
            np.testing.assert_array_almost_equal(
                result.__dict__["idx"], frame_dict[key]
            )
        elif key == "type":
            np.testing.assert_array_almost_equal(
                result.__dict__["types"], frame_dict[key]
            )
        else:
            np.testing.assert_array_almost_equal(
                result.__dict__[key], frame_dict[key]
            )


def test_lammps_raises():
    """Test the ValueError raise of write .lammpstrj file."""
    flmp = TEST_DATA_PATH / "exma_ref.lammpstrj"
    with pytest.raises(EOFError):
        with exma.io.reader.LAMMPS(flmp) as rlmp:
            rlmp.read_frame()
            rlmp.read_frame()
