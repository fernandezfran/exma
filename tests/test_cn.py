#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ======================================================================
# IMPORTS
# ======================================================================

import exma.positions
import exma.cn

import numpy as np

# ======================================================================
# TESTS
# ======================================================================


def test_monoatomic():
    """Test the coordination number of a monoatomic simple cubic crystal."""
    natoms = 27
    size = np.array([1.0, 1.0, 1.0])
    rcut = 0.4

    particles = exma.positions.Positions(natoms, size[0]).sc()
    xyz = np.concatenate((particles["x"], particles["y"], particles["z"]))

    mono = exma.cn.monoatomic(natoms, rcut)
    mono.accumulate(size, xyz)
    result = mono.end(0, xyz, writes=False)

    cnref = np.full(natoms, 6.0)

    np.testing.assert_array_equal(result, cnref)


def test_diatomic():
    """Test the coordination number of diatomic body-centered cubic crystal."""
    natoms = 54
    size = np.array([1.0, 1.0, 1.0])
    rcut = 0.3

    type1 = np.full(np.intc(natoms / 2), 1)
    type2 = np.full(np.intc(natoms / 2), 2)
    types = np.concatenate((type1, type2))
    particles = exma.positions.Positions(natoms, size[0]).bcc()
    xyz = np.concatenate((particles["x"], particles["y"], particles["z"]))

    di = exma.cn.diatomic(natoms, types, 1, 2, rcut)
    di.accumulate(size, types, xyz)
    result = di.end(types, xyz, writes=False)

    cnref = np.full(np.intc(natoms / 2), 8.0)

    np.testing.assert_array_equal(result, cnref)
