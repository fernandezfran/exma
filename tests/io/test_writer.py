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

import exma.io.writer

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
            "exma_test1.xyz",
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
            "exma_test2.xyz",
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
            "exma_test3.xyz",
            "image",
        ),
    ],
)
def test_XYZ(traj_dict, fname, ftype):
    """Test the write of an xyz file."""
    fxyz = TEST_DATA_PATH / fname

    with exma.io.writer.XYZ(fxyz, ftype) as wxyz:
        wxyz.write_frame(traj_dict)

    with open(fxyz, "r") as f:
        writed = f.read()

    with open(TEST_DATA_PATH / fname.replace("test", "ref"), "r") as f:
        expected = f.read()

    os.remove(fxyz)

    assert writed == expected


def test_XYZ_ValueError_raise():
    """Test the ValueError raise of write xyz file."""
    fxyz = TEST_DATA_PATH / "exma_test1.xyz"

    with pytest.raises(ValueError):
        exma.io.writer.XYZ(fxyz, "error")


@pytest.mark.parametrize(
    ("fname", "frame_dict"),
    [
        (
            "exma_test1.lammpstrj",
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
            "exma_test2.lammpstrj",
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
            "exma_test3.lammpstrj",
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
            "exma_test4.lammpstrj",
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
    """Test the write of an .lammpstrj file."""
    flmp = TEST_DATA_PATH / fname

    with exma.io.writer.LAMMPS(flmp) as wlmp:
        wlmp.write_frame(frame_dict)

    with open(flmp, "r") as f:
        writed = f.read()

    with open(TEST_DATA_PATH / fname.replace("test", "ref"), "r") as f:
        expected = f.read()

    os.remove(flmp)

    assert writed == expected


def test_in_lammps():
    """Test input files of lammps."""
    frame = {
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
    }
    fin = TEST_DATA_PATH / "exma_in.test"
    exma.io.writer.in_lammps(fin, frame)

    with open(fin, "r") as f:
        assert f.readline() == "# the first three lines are comments...\n"
        assert f.readline() == "# columns in order: id type q x y z ix iy iz\n"
        assert f.readline() == "# input file for LAMMPS generated by exma\n"
        assert f.readline() == "5 atoms\n"
        assert f.readline() == "2 atom types\n"
        assert f.readline() == "\n"
        assert f.readline() == "0.0 \t 4.500000e+00 \t xlo xhi\n"
        assert f.readline() == "0.0 \t 1.000000e+00 \t ylo yhi\n"
        assert f.readline() == "0.0 \t 6.000000e+00 \t zlo zhi\n"
        assert f.readline() == "\n"
        assert f.readline() == "Atoms\n"
        assert f.readline() == "\n"

    os.remove(fin)
