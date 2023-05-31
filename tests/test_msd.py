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

import exma.msd
from exma import read_xyz

from matplotlib.testing.decorators import check_figures_equal

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
    ("fname", "box", "msd_res"),
    [
        (
            "liquid.xyz",
            np.full(3, 8.54988),
            np.array(
                [
                    0.0,
                    0.18579898,
                    0.35747746,
                    0.5426987,
                    0.73278261,
                    0.9338407,
                    1.13437932,
                    1.36205518,
                    1.56029879,
                    1.73697416,
                    1.96884115,
                    2.18462587,
                    2.37341302,
                    2.57218206,
                    2.82805379,
                    3.12232283,
                    3.29623498,
                    3.45791592,
                    3.65135412,
                    3.87218877,
                    4.09262591,
                    4.40193404,
                    4.63385581,
                    4.80765401,
                    5.05222344,
                    5.22308566,
                    5.40990772,
                    5.67813274,
                    5.81758522,
                    6.0462092,
                    6.26295127,
                    6.41525821,
                    6.48066494,
                    6.78124055,
                    6.95534262,
                    7.22674676,
                    7.36488895,
                    7.52730555,
                    7.89060876,
                    8.13552497,
                    8.33028478,
                    8.56743357,
                    8.8541472,
                    9.01204869,
                    9.2139022,
                    9.36187584,
                    9.43919832,
                    9.80264101,
                    10.15579164,
                    10.38198812,
                    10.69345633,
                    10.90015012,
                    11.11918441,
                    11.32155051,
                    11.56063097,
                    11.769107,
                    11.99279092,
                    12.09625331,
                    12.3371963,
                    12.55848919,
                    12.5981481,
                    12.73328404,
                    12.95890588,
                    13.17773588,
                    13.37676504,
                    13.61786791,
                    13.88813064,
                    14.15893945,
                    14.33814374,
                    14.41392676,
                    14.53318692,
                    14.72803508,
                    14.74063587,
                    14.82099437,
                    14.99495963,
                    15.16357773,
                    15.45726625,
                    15.71460651,
                    16.1032025,
                    16.36619087,
                    16.46001708,
                    16.54509427,
                    16.62046373,
                    16.92323184,
                    17.15111028,
                    17.32838805,
                    17.44211325,
                    17.67580442,
                    18.01044074,
                    17.99441144,
                    18.2475785,
                    18.56229178,
                    18.8592032,
                    19.1061905,
                    19.03559045,
                    19.1955458,
                    19.29038906,
                    19.41953264,
                    19.56530818,
                    19.98805892,
                    20.49746008,
                    20.83550011,
                    20.98118283,
                    21.2237299,
                    21.5608569,
                    21.72466822,
                    21.96838882,
                    22.38860688,
                    22.60096692,
                    22.86585386,
                    23.07524908,
                    23.25634218,
                    23.33714576,
                    23.34287702,
                    23.70998703,
                    23.81080757,
                    23.90971071,
                    24.19646691,
                    24.43037671,
                    24.32978941,
                    24.35533812,
                    24.57236711,
                    24.7618253,
                    25.11521481,
                    25.28103188,
                    25.30450023,
                    25.48309271,
                    25.55583137,
                    25.55879713,
                    25.68217475,
                    25.73247962,
                    26.00844048,
                    26.06215095,
                    26.26288579,
                    26.40935604,
                    26.56342122,
                    26.65723871,
                    26.6508749,
                    26.79690477,
                    26.81037309,
                    26.95303199,
                    27.15802999,
                    27.46386169,
                    27.52098757,
                    27.66517052,
                    27.97374278,
                    28.08132325,
                    28.17014195,
                    28.34778873,
                    28.43299574,
                    28.65420689,
                    28.88235334,
                    29.37324761,
                    29.72699006,
                    29.86516599,
                    30.09353786,
                    30.49408872,
                    31.00501446,
                    31.24532821,
                    31.57681322,
                    31.99763296,
                    32.1887299,
                    32.62616242,
                    32.57142794,
                    32.84459969,
                    32.90820018,
                    33.1109161,
                    33.47233807,
                    33.74549947,
                    33.9370431,
                    34.11598648,
                    34.41971601,
                    34.41428703,
                    34.24803916,
                    34.52027277,
                    34.44971153,
                    34.52997185,
                    34.79932404,
                    34.89158712,
                    35.24137191,
                    35.2698819,
                    35.59213711,
                    35.74307491,
                    35.77802544,
                    36.23085136,
                    36.31178006,
                    36.48317429,
                    36.65046282,
                    36.77030673,
                    37.18797419,
                    37.51268828,
                    37.82048554,
                    37.87564568,
                    38.0113201,
                    38.13147069,
                    38.33394108,
                    38.62516787,
                    38.89001971,
                    38.96842628,
                    39.35482202,
                ]
            ),
        ),
        (
            "solid.xyz",
            np.full(3, 7.46901),
            np.array(
                [
                    0.0,
                    0.01437169,
                    0.01288304,
                    0.01461853,
                    0.014055,
                    0.01436555,
                    0.01421645,
                    0.01336308,
                    0.01382159,
                    0.0136442,
                    0.01449362,
                    0.01309846,
                    0.01398573,
                    0.01405019,
                    0.01507535,
                    0.01424186,
                    0.01437941,
                    0.01383886,
                    0.01296172,
                    0.0146991,
                    0.01410612,
                    0.01481501,
                    0.01420325,
                    0.01434715,
                    0.01298786,
                    0.0148253,
                    0.01423823,
                    0.01497784,
                    0.01396886,
                    0.0141679,
                    0.01425936,
                    0.01398268,
                    0.01438084,
                    0.01285733,
                    0.01322493,
                    0.01321683,
                    0.01465658,
                    0.01381374,
                    0.0139052,
                    0.01440455,
                    0.01427103,
                    0.01452457,
                    0.01281022,
                    0.01410243,
                    0.01369492,
                    0.0139859,
                    0.01327653,
                    0.0131189,
                    0.01415931,
                    0.01397922,
                    0.01421374,
                    0.01366827,
                    0.01448485,
                    0.01406899,
                    0.01499732,
                    0.01318651,
                    0.0149141,
                    0.01255177,
                    0.01353754,
                    0.01447474,
                    0.01426624,
                    0.01414591,
                    0.0144417,
                    0.01536105,
                    0.0139862,
                    0.0148494,
                    0.01352506,
                    0.01478536,
                    0.01439927,
                    0.01487077,
                    0.01435789,
                    0.01468788,
                    0.01386155,
                    0.01429983,
                    0.01342202,
                    0.01332188,
                    0.01459281,
                    0.01363498,
                    0.01397991,
                    0.01491391,
                    0.01363593,
                    0.01526958,
                    0.01373997,
                    0.01340737,
                    0.0140936,
                    0.01485331,
                    0.01458504,
                    0.01362609,
                    0.01356049,
                    0.01239612,
                    0.01503764,
                    0.01230978,
                    0.01504691,
                    0.01367651,
                    0.0133646,
                    0.01334921,
                    0.0145113,
                    0.01302501,
                    0.01427225,
                    0.0149299,
                    0.01392079,
                    0.01399918,
                    0.01425474,
                    0.01411208,
                    0.01361066,
                    0.01348393,
                    0.01363011,
                    0.01430216,
                    0.01411404,
                    0.01357662,
                    0.01538727,
                    0.01544303,
                    0.01347962,
                    0.01274687,
                    0.01343127,
                    0.01422597,
                    0.0142371,
                    0.01422791,
                    0.01391627,
                    0.01328601,
                    0.01264072,
                    0.01469106,
                    0.01274253,
                    0.01429073,
                    0.01421096,
                    0.01483002,
                    0.01406607,
                    0.01658518,
                    0.01354081,
                    0.01555258,
                    0.0134259,
                    0.01404076,
                    0.01393461,
                    0.01402499,
                    0.01449144,
                    0.01331841,
                    0.0146355,
                    0.01349181,
                    0.0141567,
                    0.01377857,
                    0.0139062,
                    0.01409331,
                    0.01437163,
                    0.01348469,
                    0.01335739,
                    0.01291143,
                    0.01308333,
                    0.01382893,
                    0.01263767,
                    0.01419714,
                    0.01282079,
                    0.01327358,
                    0.01457155,
                    0.01308606,
                    0.01339319,
                    0.01448804,
                    0.01351814,
                    0.01612279,
                    0.01282845,
                    0.01386472,
                    0.01278362,
                    0.01353139,
                    0.01342051,
                    0.0134796,
                    0.01385763,
                    0.01437814,
                    0.0144604,
                    0.01320453,
                    0.01648608,
                    0.01361549,
                    0.01548285,
                    0.01355591,
                    0.01415444,
                    0.01443563,
                    0.0152926,
                    0.01346656,
                    0.01366887,
                    0.01392625,
                    0.01311768,
                    0.01412517,
                    0.01296469,
                    0.01479347,
                    0.01371775,
                    0.01415684,
                    0.01319275,
                    0.01345091,
                    0.01445434,
                    0.01394499,
                    0.01443271,
                    0.01384929,
                    0.01437345,
                    0.01486461,
                    0.01324921,
                    0.0140282,
                    0.01423127,
                    0.01363781,
                    0.01494658,
                    0.01464389,
                    0.01276956,
                    0.01338069,
                ]
            ),
        ),
    ],
)
def test_msd_calculate(fname, box, msd_res):
    """Test the MSD calculation in LJ liquid and solid."""
    frames = read_xyz(TEST_DATA_PATH / fname, ftype="image")
    result = exma.msd.MeanSquareDisplacement(frames, 0.005, start=1).calculate(
        box
    )

    np.testing.assert_array_almost_equal(result["t"], 0.005 * np.arange(200))
    np.testing.assert_array_almost_equal(result["msd"], msd_res, decimal=5)


@check_figures_equal(extensions=["pdf", "png"])
def test_msd_plot(fig_test, fig_ref):
    """Test the MSD plot."""
    files = ["liquid.xyz", "solid.xyz"]
    boxes = [np.full(3, 8.54988), np.full(3, 7.46901)]
    msds = []
    for fname, box in zip(files, boxes):
        frames = read_xyz(TEST_DATA_PATH / fname, ftype="image")
        msd = exma.msd.MeanSquareDisplacement(frames, 0.005, start=10, stop=20)
        msd.calculate(box)
        msds.append(msd)

    # test
    test_ax = fig_test.subplots()
    for msd in msds:
        msd.plot(ax=test_ax)

    # expected
    exp_ax = fig_ref.subplots()

    exp_ax.set_xlabel("t")
    exp_ax.set_ylabel("msd")
    for msd in msds:
        exp_ax.plot(msd.df_msd_["t"], msd.df_msd_["msd"])
