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

import exma.io

import numpy as np


# ============================================================================
# CONSTANTS
# ============================================================================

TEST_DATA_PATH = pathlib.Path(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_data")
)


# ======================================================================
# TESTS
# ======================================================================


def test_xyz2lammpstrj():
    """Test rewrite xyz into lammptrsj."""
    fi = TEST_DATA_PATH / "test_ref.xyz"
    fo = TEST_DATA_PATH / "exma_xyz2.lammpstrj"
    exma.io.xyz2lammpstrj(
        fi, fo, {"box": np.array([4.5, 1.0, 6.0]), "type": {"H": 1}}
    )

    with open(fo, "r") as f:
        assert f.readline() == "ITEM: TIMESTEP\n"
        assert f.readline() == "0\n"
        assert f.readline() == "ITEM: NUMBER OF ATOMS\n"
        assert f.readline() == "5\n"
        assert f.readline() == "ITEM: BOX BOUNDS pp pp pp\n"
        assert f.readline() == "0.0	4.500000e+00\n"
        assert f.readline() == "0.0	1.000000e+00\n"
        assert f.readline() == "0.0	6.000000e+00\n"
        assert f.readline() == "ITEM: ATOMS id type x y z\n"
        assert (
            f.readline() == "1  1  2.675830e+00  5.432000e-02  1.151450e+00\n"
        )
        assert (
            f.readline() == "2  1  9.324100e-01  8.932500e-01  2.314510e+00\n"
        )
        assert (
            f.readline() == "3  1  1.234240e+00  4.314200e-01  3.968930e+00\n"
        )
        assert (
            f.readline() == "4  1  4.426360e+00  2.345100e-01  4.969050e+00\n"
        )
        assert (
            f.readline() == "5  1  3.000230e+00  5.555600e-01  5.986930e+00\n"
        )

    os.remove(fo)


def test_lammpstrj2xyz():
    """Test rewrite lammptrsj into xyz."""
    fi = TEST_DATA_PATH / "test_ref.lammpstrj"
    fo = TEST_DATA_PATH / "exma_lammpstrj2.xyz"
    exma.io.lammpstrj2xyz(fi, fo, {1: "H", 2: "O"})

    with open(fo, "r") as f:
        assert f.readline() == "5\n"
        assert f.readline() == "\n"
        assert f.readline() == "H  2.675830e+00  5.432000e-02  1.151450e+00\n"
        assert f.readline() == "H  9.324100e-01  8.932500e-01  2.314510e+00\n"
        assert f.readline() == "H  1.234240e+00  4.314200e-01  3.968930e+00\n"
        assert f.readline() == "O  4.426360e+00  2.345100e-01  4.969050e+00\n"
        assert f.readline() == "O  3.000230e+00  5.555600e-01  5.986930e+00\n"

    os.remove(fo)
