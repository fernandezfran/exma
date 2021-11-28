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

import exma.io.reader
import exma.io.writer

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
    ("traj_dict", "ftype"),
    [
        (
            {
                "natoms": 5,
                "type": np.array(5 * ["H"]),
                "x": np.array([2.67583, 0.93241, 1.23424, 4.42636, 3.00023]),
                "y": np.array([0.05432, 0.89325, 0.43142, 0.23451, 0.55556]),
                "z": np.array([1.15145, 2.31451, 3.96893, 4.96905, 5.98693]),
            },
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
            "property",
        ),
        (
            {
                "natoms": 5,
                "type": np.array(5 * ["H"]),
                "x": np.array([2.67583, 0.93241, 1.23424, 4.42636, 3.00023]),
                "y": np.array([0.05432, 0.89325, 0.43142, 0.23451, 0.55556]),
                "z": np.array([1.15145, 2.31451, 3.96893, 4.96905, 5.98693]),
                "ix": np.array([0, 0, 1, -1, 0]),
                "iy": np.array([2, 3, 0, 0, 1]),
                "iz": np.array([-2, -1, 0, 0, 1]),
            },
            "image",
        ),
    ],
)
def test_XYZ(traj_dict, ftype):
    """Test the write and read of an xyz file."""
    fxyz = TEST_DATA_PATH / "exma_test.xyz"

    wxyz = exma.io.writer.XYZ(fxyz, ftype)
    wxyz.write_frame(traj_dict)
    wxyz.file_close()

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


@pytest.mark.parametrize(
    "frame_dict",
    [
        {
            "natoms": 5,
            "box": np.array([4.5, 1.0, 6.0]),
            "id": np.arange(1, 6),
            "type": np.array([1, 1, 1, 2, 2]),
            "x": np.array([2.67583, 0.93241, 1.23424, 4.42636, 3.00023]),
            "y": np.array([0.05432, 0.89325, 0.43142, 0.23451, 0.55556]),
            "z": np.array([1.15145, 2.31451, 3.96893, 4.96905, 5.98693]),
        },
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
    ],
)
def test_LAMMPS(frame_dict):
    """Test the write and read .lammpstrj file."""
    flmp = TEST_DATA_PATH / "exma_test.lammpstrj"

    wlmp = exma.io.writer.LAMMPS(flmp)
    wlmp.write_frame(frame_dict)
    wlmp.file_close()

    rlmp = exma.io.reader.LAMMPS(flmp)
    result = rlmp.read_frame()
    rlmp.file_close()

    for key in frame_dict.keys():
        np.testing.assert_array_almost_equal(result[key], frame_dict[key])


def test_XYZ_raises():
    """Test the raises of write and read xyz file."""
    fxyz = TEST_DATA_PATH / "exma_test.xyz"

    with pytest.raises(ValueError):
        exma.io.writer.XYZ(fxyz, "error")

    with pytest.raises(ValueError):
        exma.io.reader.XYZ(fxyz, "error")

    rxyz = exma.io.reader.XYZ(fxyz)
    rxyz.read_frame()
    with pytest.raises(EOFError):
        rxyz.read_frame()
    rxyz.file_close()

    os.remove(fxyz)


def test_LAMMPS_raises():
    """Test the raises of write and read .lammpstrj file."""
    flmp = TEST_DATA_PATH / "exma_test.lammpstrj"
    rlmp = exma.io.reader.LAMMPS(flmp)
    rlmp.read_frame()
    with pytest.raises(EOFError):
        rlmp.read_frame()
    rlmp.file_close()

    os.remove(flmp)
