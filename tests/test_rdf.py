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
import exma.rdf.gofr

import numpy as np

# ======================================================================
# TESTS
# ======================================================================

def test_monoatomic():
    """Test the RDF of a monoatomic fcc crystal."""
    rdfref_x = np.arange(0.025, 0.5, 0.05)
    rdfref_y = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            3.478797,
            0.0,
            0.835459,
            0.0,
            1.955821,
            0.78305,
        ]
    )
    rdfref = np.split(np.concatenate((rdfref_x, rdfref_y)), 2)

    natoms = 108
    box = np.array([1.0, 1.0, 1.0])

    particles = exma.positions.Positions(natoms, box[0]).fcc()
    xyz = np.concatenate((particles["x"], particles["y"], particles["z"]))

    gofr = exma.rdf.gofr.monoatomic(natoms, box, 10)
    gofr.accumulate(box, xyz)
    result = gofr.end(writes=False)

    np.testing.assert_array_almost_equal(result, rdfref)


def test_diatomic():
    """Test the RDF of a diatomic bcc crystal."""
    rdfref_x = np.arange(0.025, 0.5, 0.05)
    rdfref_y = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 6.218508, 0.0, 0.0, 0.0, 0.0]
    )
    rdfref = np.split(np.concatenate((rdfref_x, rdfref_y)), 2)

    natoms = 54
    box = np.array([1.0, 1.0, 1.0])

    type1 = np.full(np.intc(natoms / 2), 1)
    type2 = np.full(np.intc(natoms / 2), 2)
    types = np.concatenate((type1, type2))

    particles = exma.positions.Positions(natoms, box[0]).bcc()
    xyz = np.concatenate((particles["x"], particles["y"], particles["z"]))

    gofr = exma.rdf.gofr.diatomic(natoms, box, 10, 1, 2)
    gofr.accumulate(box, types, xyz)
    result = gofr.end(types, writes=False)

    np.testing.assert_array_almost_equal(result, rdfref)
