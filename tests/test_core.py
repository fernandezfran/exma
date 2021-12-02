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

import exma.core

import numpy as np

import pytest

# ============================================================================
# CONSTANTS
# ============================================================================

TEST_DATA_PATH = pathlib.Path(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "io")
)

# ============================================================================
# TESTS
# ============================================================================


def test_TrajectoryReader_raises():
    with pytest.raises(NotImplementedError):
        tr = exma.core.TrajectoryReader(
            TEST_DATA_PATH / "test_data" / "test_ref.xyz", "error"
        )
        tr.read_frame()


def test_TrajectoryWriter_raises():
    fxyz = TEST_DATA_PATH / "test_data" / "exma_test.xyz"
    with pytest.raises(NotImplementedError):
        tw = exma.core.TrajectoryWriter(fxyz, "error")
        tw.write_frame()

    os.remove(fxyz)


@pytest.mark.parametrize(
    ("arr", "res"),
    [
        (np.arange(0, 10), True),
        ([0, 0, 1, 76, np.pi, -1], False),
        (np.random.rand(100), False),
        (np.sort(np.random.rand(100)), True),
        (np.sort(np.random.rand(100, 100)), True),
    ],
)
def test__is_sorted(arr, res):
    """Test the check if an array is sorted."""
    assert exma.core._is_sorted(arr) == res


@pytest.mark.parametrize(
    ("frame", "sorted_frame"),
    [
        (
            {
                "natoms": 3,
                "id": np.array([1, 2, 3]),
                "x": np.array([0.1, 0.5, 0.7]),
            },
            {
                "natoms": 3,
                "id": np.array([1, 2, 3]),
                "x": np.array([0.1, 0.5, 0.7]),
            },
        ),
        (
            {"natoms": 3, "id": np.array([3, 2, 1]), "x": np.array([1, 2, 3])},
            {"natoms": 3, "id": np.array([1, 2, 3]), "x": np.array([3, 2, 1])},
        ),
        (
            {
                "natoms": 5,
                "id": np.array([8, 2, 4, 5, 1]),
                "x": np.array(
                    [
                        0.85150696,
                        0.07836568,
                        0.35903994,
                        0.08902467,
                        0.65905169,
                    ]
                ),
            },
            {
                "natoms": 5,
                "id": np.array([1, 2, 4, 5, 8]),
                "x": np.array(
                    [
                        0.65905169,
                        0.07836568,
                        0.35903994,
                        0.08902467,
                        0.85150696,
                    ]
                ),
            },
        ),
    ],
)
def test__sort_traj(frame, sorted_frame):
    """Test the check sorting of a traj."""
    res = exma.core._sort_traj(frame)
    for key in sorted_frame.keys():
        np.testing.assert_array_almost_equal(res[key], sorted_frame[key])
