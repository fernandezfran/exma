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

import exma.electrochemistry.statistics

from matplotlib.testing.decorators import check_figures_equal

import numpy as np

# ======================================================================
# TESTS
# ======================================================================


def test_block_average():
    """Test the estimation of an error."""
    result = exma.electrochemistry.statistics.BlockAverage(
        [3.14, 3.15, 3.13, 3.13, 3.15, 3.15, 3.16, 3.12]
    ).calculate()

    np.testing.assert_array_equal(result["data_size"], np.array([8, 4]))
    np.testing.assert_array_almost_equal(
        result["mean"], np.array([3.1412501, 3.1412501])
    )
    np.testing.assert_array_almost_equal(
        result["var"], np.array([2.299121e-05, 2.656272e-05])
    )
    np.testing.assert_array_almost_equal(
        result["varerr"], np.array([1.228932e-05, 2.168837e-05])
    )


@check_figures_equal(extensions=["pdf", "png"])
def test_block_average_plot(fig_test, fig_ref):
    """Test the variance plot."""
    arr = np.random.rand(1000)

    block = exma.electrochemistry.statistics.BlockAverage(arr)
    result = block.calculate()

    # test
    test_ax = fig_test.subplots()
    block.plot(ax=test_ax)

    # expected
    exp_ax = fig_ref.subplots()

    exp_ax.set_xlabel("number of blocks operations")
    exp_ax.set_ylabel("block average variance")
    exp_ax.errorbar(
        list(result.index),
        np.asarray(result["var"]),
        yerr=np.asarray(result["varerr"]),
        marker="o",
        ls="",
    )


def test_block_average_save():
    """Test the save of a file."""
    block = exma.electrochemistry.statistics.BlockAverage(
        [3.14, 3.15, 3.13, 3.13, 3.15, 3.15, 3.16, 3.12]
    )
    block.calculate()
    block.save()

    with open("block_average.csv", "r") as fin:
        readed = fin.read()
    os.remove("block_average.csv")

    assert readed == (
        "data_size,mean,var,varerr\n"
        "8,3.141250e+00,2.299107e-05,1.228924e-05\n"
        "4,3.141250e+00,2.656250e-05,2.168819e-05\n"
    )
