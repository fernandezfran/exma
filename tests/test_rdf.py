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

import exma.rdf

import numpy as np

import pytest

# ============================================================================
# CONSTANTS
# ============================================================================

TEST_DATA_PATH = pathlib.Path(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_data")
)

# ======================================================================
# TESTS
# ======================================================================


@pytest.mark.parametrize(
    ("fname", "box", "rdf_res"),
    [
        (
            "liquid.xyz",
            np.full(3, 8.54988),
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.000303,
                    0.049846,
                    0.530536,
                    1.605204,
                    2.449074,
                    2.602068,
                    2.328016,
                    1.935885,
                    1.567824,
                    1.28823,
                    1.077404,
                    0.928768,
                    0.819052,
                    0.739576,
                    0.69985,
                    0.669325,
                    0.663333,
                    0.671416,
                    0.690161,
                    0.726016,
                    0.783985,
                    0.846533,
                    0.921836,
                    0.986964,
                    1.053396,
                    1.118081,
                    1.179314,
                    1.209613,
                    1.226445,
                    1.209413,
                    1.182201,
                    1.145224,
                    1.090909,
                    1.036284,
                    0.9892,
                    0.952239,
                    0.917195,
                    0.893103,
                    0.886568,
                    0.879963,
                    0.886845,
                    0.904843,
                    0.925227,
                    0.951592,
                    0.973656,
                    1.000878,
                    1.021233,
                    1.042432,
                    1.057729,
                    1.065852,
                    1.064024,
                    1.065103,
                    1.058088,
                    1.046985,
                    1.033373,
                    1.015977,
                    1.001686,
                    0.987246,
                    0.977131,
                    0.967341,
                    0.961565,
                    0.957972,
                    0.961741,
                    0.962554,
                    0.971379,
                    0.979026,
                    0.990421,
                    0.997328,
                    1.009341,
                    1.013717,
                    1.020723,
                    1.02103,
                    1.022537,
                    1.025864,
                    1.021476,
                    1.017072,
                    1.013182,
                    1.007187,
                    1.000785,
                    0.993021,
                ]
            ),
        ),
        (
            "solid.xyz",
            np.full(3, 7.46901),
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.000415,
                    0.063499,
                    0.801847,
                    3.017233,
                    5.01277,
                    4.824287,
                    3.135037,
                    1.498764,
                    0.54967,
                    0.1626,
                    0.040546,
                    0.023532,
                    0.064575,
                    0.184516,
                    0.434718,
                    0.765335,
                    1.011242,
                    0.986079,
                    0.716085,
                    0.391516,
                    0.175114,
                    0.156261,
                    0.429954,
                    1.123152,
                    2.123976,
                    2.836789,
                    2.718657,
                    1.88933,
                    0.992119,
                    0.449322,
                    0.363165,
                    0.618845,
                    0.94565,
                    1.080114,
                    0.915824,
                    0.63059,
                    0.496517,
                    0.674686,
                    1.106638,
                    1.515756,
                    1.606366,
                    1.287405,
                    0.812412,
                    0.493543,
                    0.420703,
                    0.471396,
                    0.502123,
                    0.518806,
                    0.69135,
                    1.198601,
                    1.90146,
                    2.350094,
                    2.194164,
                    1.564477,
                    0.907838,
                    0.50883,
                    0.364095,
                    0.360827,
                    0.474226,
                    0.744384,
                    1.114382,
                    1.36686,
                    1.354731,
                    1.147148,
                    0.974079,
                    0.93332,
                    0.917362,
                    0.853337,
                    0.795278,
                    0.815541,
                    0.857573,
                    0.838587,
                    0.779574,
                    0.79139,
                    0.879244,
                    1.032008,
                    1.29641,
                ]
            ),
        ),
    ],
)
def test_RadialDistributionFunction_calculate(fname, box, rdf_res):
    """Test the RDF calculation in LJ liquid and solid."""
    nbin = 100
    rmax = np.min(box) / 2
    result = exma.rdf.RadialDistributionFunction(
        str(TEST_DATA_PATH / fname), "Ar", "Ar", start=1, nbin=nbin, rmax=rmax
    ).calculate(box)

    dr = rmax / nbin
    np.testing.assert_array_almost_equal(
        result["r"], np.arange(dr / 2, rmax + dr / 2, dr)
    )
    np.testing.assert_array_almost_equal(result["rdf"], rdf_res)


@pytest.mark.parametrize(
    "fname",
    ["liquid.out", "solid.xtc", "gas.lammsptrj", "dump.asd.123.lammptsrj"],
)
def test_RadialDistributionFunction_raises(fname):
    """Test the RDF ValueError raise."""
    with pytest.raises(ValueError):
        exma.rdf.RadialDistributionFunction(fname, "H", "H").calculate()


@pytest.mark.parametrize(
    ("fname", "box"),
    [("liquid.xyz", np.full(3, 8.54988)), ("solid.xyz", np.full(3, 7.46901))],
)
def test_RadialDistributionFunction_warning(fname, box):
    """Test the RDF EOF warning."""
    with pytest.warns(UserWarning):
        exma.rdf.RadialDistributionFunction(
            str(TEST_DATA_PATH / fname),
            "Ar",
            "Ar",
            start=190,
            stop=210,
            step=5,
        ).calculate(box)


def test_RadialDistributionFunction_plot():
    """Test the RDF plot."""
    with pytest.raises(NotImplementedError):
        exma.rdf.RadialDistributionFunction("something.xyz", "H", "H").plot()


def test_RadialDistributionFunction_save():
    """Test the RDF save."""
    with pytest.raises(NotImplementedError):
        exma.rdf.RadialDistributionFunction("something.xyz", "H", "H").save()
