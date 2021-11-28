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


def _is_sorted(idx):
    """Tells if the array x is sorted (-> True) or not (-> False)."""
    return (np.diff(idx) >= 0).all()


def _sort_traj(frame, dontsort=("natoms", "box")):
    """Sort all the traj from the sortening of the atoms id."""
    id_argsort = np.argsort(frame["id"])

    for key in frame.keys():
        frame[key] = (
            frame[key][id_argsort] if key not in dontsort else frame[key]
        )
    return frame
