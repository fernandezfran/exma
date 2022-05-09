#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""An estimator of the error of a serie of data.

It is intended to facilitate the estimation of the error of thermodynamic
quantities before using them to calculate electrochemical properties.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

# ============================================================================
# FUNCTIONS
# ============================================================================


class BlockAverage:
    """Estimating error method when data are correlated.

    This method offers an easy and efficient way to estimate the error of
    correlated data by a renormalization groups, as described by
    H. Flyvbjerg and H. G. Petersen [1]_.

    Parameters
    ----------
    x : np.array
        where the time series is

    References
    ----------
    .. [1] Flyvbjerg, H. and Petersen, H.G., 1989. Error estimates on averages
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
            A `pd.DataFrame` with `data_size`, `mean`, `var` and `varerr`
            as columns that gives information about the data size changes,
            the mean value of each block, the corresponding variance and the
            error of that variance, respectively.
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

        self.df_ = pd.DataFrame(
            data={
                "data_size": np.array(data_size),
                "mean": np.array(mean),
                "var": np.array(var),
                "varerr": np.array(varerr),
            }
        )

        return self.df_

    def plot(self, ax=None, errorbar_kws=None):
        """Flyvbjerg & Petersen plot.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axis, default=None
            current metplotlib axis

        errorbar_kws : dict, default={"marker": "o", "ls": ""}
            additional keyword arguments that are passed and are documented
            in `matplotlib.pyplot.errorbar_kws`.

        Returns
        -------
        matplotlib.pyplot.Axis
            the axis with the plot
        """
        ax = plt.gca() if ax is None else ax

        errorbar_kws = {} if errorbar_kws is None else errorbar_kws
        for key, value in zip(["marker", "ls"], ["o", ""]):
            errorbar_kws.setdefault(key, value)

        ax.set_xlabel("number of blocks operations")
        ax.set_ylabel("block average variance")

        ax.errorbar(
            list(self.df_.index),
            np.asarray(self.df_["var"]),
            yerr=np.asarray(self.df_["varerr"]),
            **errorbar_kws,
        )

        return ax
