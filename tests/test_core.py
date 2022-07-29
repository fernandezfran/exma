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

TEST_DATA_PATH = pathlib.Path((os.path.abspath(os.path.dirname(__file__))))

# ============================================================================
# TESTS
# ============================================================================


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
def test__sorted(arr, res):
    """Test the check if an array is sorted."""
    frame = exma.core.AtomicSystem(idx=arr)
    assert frame._sorted() == res


@pytest.mark.parametrize(
    ("frame", "sorted_frame"),
    [
        (
            exma.core.AtomicSystem(
                natoms=3,
                idx=np.array([1, 2, 3]),
                x=np.array([0.1, 0.5, 0.7]),
            ),
            exma.core.AtomicSystem(
                natoms=3,
                idx=np.array([1, 2, 3]),
                x=np.array([0.1, 0.5, 0.7]),
            ),
        ),
        (
            exma.core.AtomicSystem(
                natoms=3, idx=np.array([3, 2, 1]), x=np.array([1, 2, 3])
            ),
            exma.core.AtomicSystem(
                natoms=3, idx=np.array([1, 2, 3]), x=np.array([3, 2, 1])
            ),
        ),
        (
            exma.core.AtomicSystem(
                natoms=5,
                idx=np.array([8, 2, 4, 5, 1]),
                x=np.array(
                    [
                        0.85150696,
                        0.07836568,
                        0.35903994,
                        0.08902467,
                        0.65905169,
                    ]
                ),
            ),
            exma.core.AtomicSystem(
                natoms=5,
                idx=np.array([1, 2, 4, 5, 8]),
                x=np.array(
                    [
                        0.65905169,
                        0.07836568,
                        0.35903994,
                        0.08902467,
                        0.85150696,
                    ]
                ),
            ),
        ),
    ],
)
def test__sort(frame, sorted_frame):
    """Test the check sorting of a frame."""
    res = frame._sort()
    for key in sorted_frame.__dict__.keys():
        if res.__dict__[key] is not None:
            np.testing.assert_array_almost_equal(
                res.__dict__[key], sorted_frame.__dict__[key]
            )


def test_tr_raises():
    with pytest.raises(NotImplementedError):
        with exma.core.TrajectoryReader(
            TEST_DATA_PATH / "test_data" / "test_ref.xyz", "error"
        ) as tr:
            tr.read_frame()


def test_tw_raises():
    fxyz = TEST_DATA_PATH / "test_data" / "exma_test.xyz"
    with pytest.raises(NotImplementedError):
        with exma.core.TrajectoryWriter(fxyz, "error") as tw:
            tw.write_frame(0)

    os.remove(fxyz)
