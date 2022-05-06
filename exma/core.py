#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Core Atomic System class of exma."""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np


# ============================================================================
# CLASSES
# ============================================================================


class AtomicSystem:
    """Define the configurations of the atomic system.

    natoms : int, default=None
        the number of atoms in the atomic system

    box : np.array, default=None
        the box lenght in each direction

    types : np.array, default=None
        the types of the atoms

    idx : np.array, default=None
        the index of the atoms

    q : np.array, default=None
        a property of each atom in the atomic system

    x : np.array, default=None
        the positions of the atoms in the x direction

    y : np.array, default=None
        the positions of the atoms in the y direction

    z : np.array, default=None
        the positions of the atoms in the z direction

    ix : np.array, default=None
        the corresponding image of the positions of the atoms in the x
        direction

    iy : np.array, default=None
        the corresponding image of the positions of the atoms in the y
        direction

    iz : np.array, default=None
        the corresponding image of the positions of the atoms in the z
        direction
    """

    def __init__(
        self,
        natoms=None,
        box=None,
        types=None,
        idx=None,
        q=None,
        x=None,
        y=None,
        z=None,
        ix=None,
        iy=None,
        iz=None,
    ):
        self.natoms = natoms
        self.box = box

        self.idx = idx
        self.types = types

        self.q = q

        self.x = x
        self.y = y
        self.z = z

        self.ix = ix
        self.iy = iy
        self.iz = iz

    def _is_sorted(self):
        """Tells if the array x is sorted (-> True) or not (-> False)."""
        return (np.diff(self.idx) >= 0).all()

    def _sort_frame(self, dontsort=("natoms", "box")):
        """Sort the frame from the sortening of the atoms id."""
        id_argsort = np.argsort(self.idx)

        for key in self.__dict__.keys():
            if (
                key.startswith("_")
                or key in dontsort
                or self.__dict__[key] is None
            ):
                continue

            self.__dict__[key] = self.__dict__[key][id_argsort]

        return self

    def _mask_type(self, atom_type):
        """Get a masked array by an specific type of atom."""
        return self.types == atom_type

    def _natoms_type(self, mask_type):
        """Count the number of atoms of an specific type."""
        return np.count_nonzero(mask_type)
