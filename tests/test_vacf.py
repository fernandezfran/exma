#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021-2023, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ============================================================================
# IMPORTS
# ============================================================================

import os
import pathlib

import exma.vacf
from exma import AtomicSystem

from matplotlib.testing.decorators import check_figures_equal

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


def test_vacf_calculate():
    """Test the VACF calculation."""
    vacf_res = np.array([0.0961, 0.1829, 0.2511, 0.2945, 0.031])

    zero = np.zeros(1)
    frames = [
        AtomicSystem(natoms=1, vx=np.array([0.31]), vy=zero, vz=zero),
        AtomicSystem(natoms=1, vx=np.array([0.59]), vy=zero, vz=zero),
        AtomicSystem(natoms=1, vx=np.array([0.81]), vy=zero, vz=zero),
        AtomicSystem(natoms=1, vx=np.array([0.95]), vy=zero, vz=zero),
        AtomicSystem(natoms=1, vx=np.array([0.1]), vy=zero, vz=zero),
    ]
    result = exma.vacf.VelocityAutocorrelationFunction(frames, 0.2).calculate()

    np.testing.assert_array_almost_equal(
        result["t"], 0.2 * np.arange(0, len(frames))
    )
    np.testing.assert_array_almost_equal(result["vacf"], vacf_res, decimal=5)


@check_figures_equal(extensions=["pdf", "png"])
def test_vacf_plot(fig_test, fig_ref):
    """Test the VACF plot."""
    zero = np.zeros(1)
    frames = [
        AtomicSystem(natoms=1, vx=np.array([0.31]), vy=zero, vz=zero),
        AtomicSystem(natoms=1, vx=np.array([0.59]), vy=zero, vz=zero),
        AtomicSystem(natoms=1, vx=np.array([0.81]), vy=zero, vz=zero),
        AtomicSystem(natoms=1, vx=np.array([0.95]), vy=zero, vz=zero),
        AtomicSystem(natoms=1, vx=np.array([0.1]), vy=zero, vz=zero),
    ]
    vacf = exma.vacf.VelocityAutocorrelationFunction(frames, 0.2)
    vacf.calculate()

    # test
    test_ax = fig_test.subplots()
    vacf.plot(ax=test_ax)

    # expected
    exp_ax = fig_ref.subplots()

    exp_ax.set_xlabel("t")
    exp_ax.set_ylabel("vacf")
    exp_ax.plot(vacf.df_vacf_["t"], vacf.df_vacf_["vacf"])
