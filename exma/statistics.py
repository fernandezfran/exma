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


class BlockAverage:
    """Estimating error method when data are correlated.

    This method offers an easy and efficient way to estimate the error of
    correlated data by a renormalization groups, as described by
    H. Flyvbjerg and H. G. Petersen [4]_.

    Parameters
    ----------
    x : np.array
        where the time series is

    References
    ----------
    .. [4] Flyvbjerg, H. and Petersen, H.G., 1989. Error estimates on averages
       of correlated data. `The Journal of Chemical Physics`, 91(1),
       pp.461-466.
    """

    def __init__(self, x):
        self.x = x

    def calculate(self):
        """Calculate the estimation of the error.

        Returns
        -------
        pd.DataFrame
            with `data_size`, `mean`, `var` and `varerr` as columns that
            gives information about the data size changes, the mean value
            of each block, the corresponding variance and the error of
            that variance, respectively.
        """
        data_size, mean, var, varerr = [], [], [], []

        idx = 0
        data_size.append(len(self.x))
        mean.append(np.mean(self.x))
        var.append(np.var(self.x) / (data_size[idx] - 1))
        varerr.append(np.sqrt(2.0 / (data_size[idx] - 1)) * var[idx])

        oldx = self.x
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

        self.df = pd.DataFrame(
            data={
                "data_size": np.array(data_size),
                "mean": np.array(mean),
                "var": np.array(var),
                "varerr": np.array(varerr),
            }
        )

        return self.df

    def plot(self):
        """Flyvbjerg & Petersen plot.

        Not implemented yet.
        """
        raise NotImplementedError("To be implemented soon.")
