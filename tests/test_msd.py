#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ======================================================================
# IMPORTS
# ======================================================================

import exma.msd

import numpy as np

# ======================================================================
# TESTS
# ======================================================================


def test_monoatomic_wrapped():
    """Test the monoatomic wrapped mean square displacement."""
    natoms = 1
    box = np.array([2.0, 2.0, 2.0])
    xyzi = np.array([0.0, 0.0, 0.0])
    xyzf, imgf = xyzi, np.array([1, 1, 1])

    msd = exma.msd.monoatomic(natoms, box, xyzi)
    result = msd.wrapped(box, xyzf, imgf)

    msdref = np.array([1.0, 12.0])
    np.testing.assert_array_equal(result, msdref)


def test_monoatomic_unwrapped():
    """Test the monoatomic unwrapped mean square displacement."""
    natoms = 1
    box = np.array([2.0, 2.0, 2.0])
    xyzi = np.array([0.0, 0.0, 0.0])
    xyzf = np.array([1.0, 1.0, 1.0])

    msd = exma.msd.monoatomic(natoms, box, xyzi)
    result = msd.unwrapped(xyzf)

    msdref = np.array([1.0, 3.0])
    np.testing.assert_array_equal(result, msdref)


def test_diatomic_wrapped():
    """Test the diatomic wrapped mean square displacement."""
    # two particles: one in the origin of the box (type 1) and the other
    # int the center of the box (type 2)
    natoms = 2
    box = np.array([2.0, 2.0, 2.0])
    types = np.array([1, 2])
    xyzi = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    xyzf = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    imgf = np.array([0, 1, 0, 1, 0, 1])

    msd = exma.msd.diatomic(natoms, box, types, xyzi, 1, 2)
    result = msd.wrapped(box, types, xyzf, imgf)

    msdref = np.array([1.0, 3.0, 3.0, 3.0])
    np.testing.assert_array_equal(result, msdref)


def test_diatomic_unwrapped():
    """Test the diatomic unwrapped mean square displacement."""
    # two particles: one in the origin of the box (type 1) and the other
    # int the center of the box (type 2)
    natoms = 2
    box = np.array([2.0, 2.0, 2.0])
    types = np.array([1, 2])
    xyzi = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    xyzf = np.array([0.0, 1.5, 0.0, 1.5, 0.0, 1.5])

    msd = exma.msd.diatomic(natoms, box, types, xyzi, 1, 2)
    result = msd.unwrapped(types, xyzf)

    msdref = np.array([1.0, 0.0, 0.75, 0.375])
    np.testing.assert_array_equal(result, msdref)
