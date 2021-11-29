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
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    3.02911866e-04,
                    4.98463643e-02,
                    5.30536035e-01,
                    1.60520391e00,
                    2.44907379e00,
                    2.60206830e00,
                    2.32801608e00,
                    1.93588525e00,
                    1.56782444e00,
                    1.28822973e00,
                    1.07740426e00,
                    9.28767729e-01,
                    8.19051816e-01,
                    7.39598267e-01,
                    6.99828931e-01,
                    6.69324619e-01,
                    6.63333275e-01,
                    6.71415819e-01,
                    6.90160540e-01,
                    7.26032142e-01,
                    7.83985413e-01,
                    8.46518252e-01,
                    9.21835769e-01,
                    9.86963621e-01,
                    1.05339623e00,
                    1.11808103e00,
                    1.17931384e00,
                    1.20961331e00,
                    1.22644524e00,
                    1.20941302e00,
                    1.18220067e00,
                    1.14522433e00,
                    1.09090869e00,
                    1.03628384e00,
                    9.89200392e-01,
                    9.52238784e-01,
                    9.17195130e-01,
                    8.93103282e-01,
                    8.86567596e-01,
                    8.79963050e-01,
                    8.86845374e-01,
                    9.04842769e-01,
                    9.25226530e-01,
                    9.51592102e-01,
                    9.73662531e-01,
                    1.00088440e00,
                    1.02122698e00,
                    1.04242634e00,
                    1.05773401e00,
                    1.06585728e00,
                    1.06402386e00,
                    1.06510303e00,
                    1.05809282e00,
                    1.04698047e00,
                    1.03337715e00,
                    1.01596787e00,
                    1.00168569e00,
                    9.87250681e-01,
                    9.77134730e-01,
                    9.67333017e-01,
                    9.61576492e-01,
                    9.57975542e-01,
                    9.61733281e-01,
                    9.62542800e-01,
                    9.71379359e-01,
                    9.79026055e-01,
                    9.90421204e-01,
                    9.97327554e-01,
                    1.00934078e00,
                    1.01371718e00,
                    1.02072346e00,
                    1.02102994e00,
                    1.02253700e00,
                    1.02586396e00,
                    1.02147640e00,
                    1.01707181e00,
                    1.01318184e00,
                    1.00718744e00,
                    1.00078544e00,
                    9.93021388e-01,
                ]
            ),
        ),
        (
            "solid.xyz",
            np.full(3, 7.46901),
            np.array(
                [
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    4.14936243e-04,
                    6.34993759e-02,
                    8.01847239e-01,
                    3.01723328e00,
                    5.01277046e00,
                    4.82428739e00,
                    3.13503663e00,
                    1.49876398e00,
                    5.49669957e-01,
                    1.62600203e-01,
                    4.05455135e-02,
                    2.35322907e-02,
                    6.45745610e-02,
                    1.84516206e-01,
                    4.34718203e-01,
                    7.65335214e-01,
                    1.01124215e00,
                    9.86078878e-01,
                    7.16084508e-01,
                    3.91515873e-01,
                    1.75113611e-01,
                    1.56260632e-01,
                    4.29953581e-01,
                    1.12315155e00,
                    2.12397574e00,
                    2.83678925e00,
                    2.71865655e00,
                    1.88933039e00,
                    9.92118750e-01,
                    4.49321601e-01,
                    3.63164631e-01,
                    6.18845000e-01,
                    9.45650069e-01,
                    1.08011449e00,
                    9.15823657e-01,
                    6.30590269e-01,
                    4.96516669e-01,
                    6.74685936e-01,
                    1.10663783e00,
                    1.51575588e00,
                    1.60636583e00,
                    1.28740524e00,
                    8.12412101e-01,
                    4.93543233e-01,
                    4.20702788e-01,
                    4.71396206e-01,
                    5.02122588e-01,
                    5.18806476e-01,
                    6.91349896e-01,
                    1.19860053e00,
                    1.90145957e00,
                    2.35009388e00,
                    2.19416378e00,
                    1.56447686e00,
                    9.07837673e-01,
                    5.08830242e-01,
                    3.64095073e-01,
                    3.60826957e-01,
                    4.74226301e-01,
                    7.44384014e-01,
                    1.11438196e00,
                    1.36686048e00,
                    1.35473129e00,
                    1.14714759e00,
                    9.74078973e-01,
                    9.33320296e-01,
                    9.17361913e-01,
                    8.53337362e-01,
                    7.95277677e-01,
                    8.15540955e-01,
                    8.57573220e-01,
                    8.38587282e-01,
                    7.79573846e-01,
                    7.91390135e-01,
                    8.79243807e-01,
                    1.03200774e00,
                    1.29641001e00,
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
