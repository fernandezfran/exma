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

import exma.io

import numpy as np


# ============================================================================
# CONSTANTS
# ============================================================================

TEST_DATA_PATH = pathlib.Path(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_data")
)


# ============================================================================
# TESTS
# ============================================================================


def test_xyz2lammpstrj():
    """Test rewrite xyz into lammpstrj."""
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


def test_xyz2inlmp():
    """Test the write of an xyz frame to lammps input data file."""
    fi = TEST_DATA_PATH / "test_ref.xyz"
    fo = TEST_DATA_PATH / "exma_in.test"
    exma.io.xyz2inlmp(
        fi,
        fo,
        {"box": np.array([4.5, 1.0, 6.0], dtype=np.float32), "type": {"H": 1}},
    )

    with open(fo, "r") as f:
        assert f.readline() == "# the first three lines are comments...\n"
        assert f.readline() == "# columns in order: id type x y z\n"
        assert f.readline() == "# input file for LAMMPS generated by exma\n"
        assert f.readline() == "5 atoms\n"
        assert f.readline() == "1 atom types\n"
        assert f.readline() == "\n"
        assert f.readline() == "0.0 \t 4.500000e+00 \t xlo xhi\n"
        assert f.readline() == "0.0 \t 1.000000e+00 \t ylo yhi\n"
        assert f.readline() == "0.0 \t 6.000000e+00 \t zlo zhi\n"
        assert f.readline() == "\n"
        assert f.readline() == "Atoms\n"
        assert f.readline() == "\n"
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


def test_xyz2inlmp_if():
    """Test the write of an xyz frame to lammps input data file."""
    fi = TEST_DATA_PATH / "test_ref.xyz"
    fo = TEST_DATA_PATH / "exma_in.test"
    exma.io.xyz2inlmp(
        fi,
        fo,
        {
            "box": np.array([4.5, 1.0, 6.0], dtype=np.float32),
            "type": {"H": 1},
            "q": np.zeros(5),
        },
    )

    with open(fo, "r") as f:
        assert f.readline() == "# the first three lines are comments...\n"
        assert f.readline() == "# columns in order: id type q x y z\n"
        assert f.readline() == "# input file for LAMMPS generated by exma\n"
        assert f.readline() == "5 atoms\n"
        assert f.readline() == "1 atom types\n"
        assert f.readline() == "\n"
        assert f.readline() == "0.0 \t 4.500000e+00 \t xlo xhi\n"
        assert f.readline() == "0.0 \t 1.000000e+00 \t ylo yhi\n"
        assert f.readline() == "0.0 \t 6.000000e+00 \t zlo zhi\n"
        assert f.readline() == "\n"
        assert f.readline() == "Atoms\n"
        assert f.readline() == "\n"
        assert (
            f.readline()
            == "1  1  0.0  2.675830e+00  5.432000e-02  1.151450e+00\n"
        )
        assert (
            f.readline()
            == "2  1  0.0  9.324100e-01  8.932500e-01  2.314510e+00\n"
        )
        assert (
            f.readline()
            == "3  1  0.0  1.234240e+00  4.314200e-01  3.968930e+00\n"
        )
        assert (
            f.readline()
            == "4  1  0.0  4.426360e+00  2.345100e-01  4.969050e+00\n"
        )
        assert (
            f.readline()
            == "5  1  0.0  3.000230e+00  5.555600e-01  5.986930e+00\n"
        )

    os.remove(fo)


def test_lammpstrj2xyz():
    """Test rewrite lammpstrj into xyz."""
    fi = TEST_DATA_PATH / "exma_ref.lammpstrj"
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


def test_lammpstrj2inlmp():
    """Test the write of an xyz frame to lammps input data file."""
    fi = TEST_DATA_PATH / "exma_ref.lammpstrj"
    fo = TEST_DATA_PATH / "exma_in.test"
    exma.io.lammpstrj2inlmp(fi, fo)

    with open(fo, "r") as f:
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
        assert f.readline() == (
            "1  1  -3.356000e-01  2.675830e+00  5.432000e-02  "
            "1.151450e+00  0  2  -2\n"
        )
        assert f.readline() == (
            "2  1  -3.263600e-01  9.324100e-01  8.932500e-01  "
            "2.314510e+00  0  3  -1\n"
        )
        assert f.readline() == (
            "3  1  -3.425600e-01  1.234240e+00  4.314200e-01  "
            "3.968930e+00  1  0  0\n"
        )
        assert f.readline() == (
            "4  2  5.436500e-01  4.426360e+00  2.345100e-01  "
            "4.969050e+00  -1  0  0\n"
        )
        assert f.readline() == (
            "5  2  4.646300e-01  3.000230e+00  5.555600e-01  "
            "5.986930e+00  0  1  1\n"
        )

    os.remove(fo)
