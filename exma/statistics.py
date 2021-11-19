#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""An estimator of the error of a serie of data."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pandas as pd

# ============================================================================
# FUNCTIONS
# ============================================================================


def block_average(x):
    """A method for estimating errors when data are correlated.

    See: H. Flyvbjerg and H. G. Petersen: Averages of correlated data (1989)

    Parameters
    ----------
    x : array
        where the time series is

    Returns
    -------
    ``pd.DataFrame``
        with `idx`, `data_size`, `mean`, `var` and `varerr` as columns that
        gives information about the number of times that the block sums were
        applied, the data size changes, the mean value of each block, the
        corresponding variance and the error of that variance, respectively.
    """
    data_size, mean, var, varerr = [], [], [], []

    idx = 0
    data_size.append(len(x))
    mean.append(np.mean(x))
    var.append(np.var(x) / (data_size[idx] - 1))
    varerr.append(np.sqrt(2.0 / (data_size[idx] - 1)) * var[idx])

    oldx = x
    while np.intc(len(oldx) / 2) > 2:
        newx = np.zeros(np.intc(len(oldx) / 2))

        for k in range(len(newx)):
            newx[k] = 0.5 * (oldx[2 * k - 1] + oldx[2 * k])

        idx += 1
        data_size.append(len(newx))
        mean.append(np.mean(newx))
        var.append(np.var(newx) / (data_size[idx] - 1))
        varerr.append(np.sqrt(2.0 / (data_size[idx] - 1)) * var[idx])

        oldx = newx

    return pd.DataFrame(
        data={
            "idx": np.arange(idx + 1),
            "data_size": np.array(data_size),
            "mean": np.array(mean),
            "var": np.array(var),
            "varerr": np.array(varerr),
        }
    )
