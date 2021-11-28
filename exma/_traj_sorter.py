#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ======================================================================
# DOCS
# ======================================================================

"""Functions to sort trajectory."""

# ======================================================================
# IMPORTS
# ======================================================================

import numpy as np

# ======================================================================
# FUNCTIONS
# ======================================================================


def _is_sorted(x):
    """Tells if the array x is sorted (-> True) or not (-> False).

    `x` is usually the frame['id'] for .lammpstrj files.
    """
    return (np.diff(x) >= 0).all()


def _sort_traj(frame):
    """Sort all the traj from the sortening of the atoms id."""
    id_argsort = np.argsort(frame["id"])
    for key in frame.keys():
        if key in ["natoms", "box"]:
            continue
        frame[key] = frame[key][id_argsort]
    return frame
