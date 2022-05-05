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

import exma.io._rw

import pytest

# ============================================================================
# CONSTANTS
# ============================================================================

TEST_DATA_PATH = pathlib.Path((os.path.abspath(os.path.dirname(__file__))))

# ============================================================================
# TESTS
# ============================================================================


def test_TrajectoryReader_raises():
    with pytest.raises(NotImplementedError):
        with exma.io._rw.TrajectoryReader(
            TEST_DATA_PATH / "test_data" / "test_ref.xyz", "error"
        ) as tr:
            tr.read_frame()


def test_TrajectoryWriter_raises():
    fxyz = TEST_DATA_PATH / "test_data" / "exma_test.xyz"
    with pytest.raises(NotImplementedError):
        with exma.io._rw.TrajectoryWriter(fxyz, "error") as tw:
            tw.write_frame()

    os.remove(fxyz)
