#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ======================================================================
# DOCS
# ======================================================================

"""Implementation of Mean Square Displacement."""

# ======================================================================
# IMPORTS
# ======================================================================

import numpy as np

# ======================================================================
# CLASSES
# ======================================================================


class msd:
    """Mean Square Displacement.

    Notes
    -----
    Remember that trajectories must be sorted with the same order as
    reference positions (no problem with .xyz files, but with .lammpstrj
    file a np.sort / np.argsort must be used before the calculation).
    """


class monoatomic(msd):
    """MSD of a monoatomic system.

    Parameters
    ----------
    natoms : int
        the number of atoms

    box_size : np.array
        with the box lenght in x, y, z

    x_ref : np.array
        the reference positions in the SoA convention (i.e. first all the x,
        then y and then z)

    image_ref : np.array, default=None
        reference image, same as positions
    """

    def __init__(self, natoms, box_size, x_ref, image_ref=None):

        self.natoms = natoms

        if image_ref is not None:
            x, y, z = np.split(x_ref, 3)
            ix, iy, iz = np.split(image_ref, 3)
            x += box_size[0] * ix
            y += box_size[1] * iy
            z += box_size[2] * iz
            x_ref = np.concatenate((x, y, z))

        self.ref = np.split(x_ref, 3)
        self.frame_ = 0

    def wrapped(self, box_size, positions, image):
        """If trajectory is wrapped inside the simulation box.

        Parameters
        ----------
        box_size : np.array
            with the box lenght in x, y, z

        positions : np.array
            the positions in the SoA convention (i.e. first all the x,
            then y and then z)

        image : np.array
            same as positions

        Returns
        -------
        np.array
            with the frame in the first value and the msd in the second
        """
        msd = 0.0
        positions = np.split(positions, 3)
        image = np.split(image, 3)
        meansd = np.zeros(self.natoms, dtype=np.float32)
        for i in range(3):
            xx = positions[i] + image[i] * box_size[i] - self.ref[i]
            meansd += xx * xx
        msd = np.sum(meansd) / self.natoms

        self.frame_ += 1

        return np.array([self.frame_, msd], dtype=np.float32)

    def unwrapped(self, positions):
        """If the trajectory is unwrapped outside of the simulation box.

        Parameters
        ----------
        positions : np.array
            the positions in the SoA convention (i.e. first all the x,
            then y and then z)

        Returns
        -------
        np.array
            with the frame in the first value and the msd in the second
        """
        msd = 0.0
        positions = np.split(positions, 3)
        meansd = np.zeros(self.natoms, dtype=np.float32)
        for i in range(3):
            xx = positions[i] - self.ref[i]
            meansd += xx * xx
        msd = np.sum(meansd) / self.natoms

        self.frame_ += 1

        return np.array([self.frame_, msd], dtype=np.float32)


class diatomic(msd):
    """MSD of a diatomic system.

    Parameters
    ----------
    natoms : int
        the number of atoms in the frame

    box_size : np.array
        with the box lenght in x, y, z

    atom_type : list
        the type of the atoms

    x_ref : np.array
        the reference positions in the SoA convention (i.e. first all
        the x, then y and then z)

    atom_type_a : int
        one type of atom

    atom_type_b : int
        another type of atom

    image_ref : np.array, default=None
        reference image, same as positions
    """

    def __init__(
        self,
        natoms,
        box_size,
        atom_type,
        x_ref,
        atom_type_a,
        atom_type_b,
        image_ref=None,
    ):
        self.natoms = natoms

        if image_ref is not None:
            x, y, z = np.split(x_ref, 3)
            ix, iy, iz = np.split(image_ref, 3)
            x += box_size[0] * ix
            y += box_size[1] * iy
            z += box_size[2] * iz
            x_ref = np.concatenate((x, y, z))

        self.ref = np.split(x_ref, 3)
        self.atom_type_a = atom_type_a
        self.atom_type_b = atom_type_b

        self.frame_ = 0
        self.n_a_ = np.count_nonzero(atom_type == atom_type_a)
        self.n_b_ = np.count_nonzero(atom_type == atom_type_b)

    def wrapped(self, box_size, atom_type, positions, image):
        """If trajectory is wrapped inside the simulation box.

        Parameters
        ----------
        box_size : np.array
            with the box lenght in x, y, z

        atom_type : list
            the type of the atoms

        positions : np.array
            the positions in the SoA convention (i.e. first all the x,
            then y and then z)

        image : np.array
            same as positions

        Returns
        -------
        np.array
            with the frame in the first value, the msd of atom type a in the
            second, the msd of atom type b in the third and the total msd in
            the fourth.
        """
        msd_a, msd_b, msd_t = 0.0, 0.0, 0.0

        positions = np.split(positions, 3)
        image = np.split(image, 3)
        meansd = np.zeros(self.natoms, dtype=np.float32)
        for i in range(3):
            xx = positions[i] + image[i] * box_size[i] - self.ref[i]
            meansd += xx * xx

        msd_t = np.sum(meansd) / self.natoms
        msd_a = np.sum(meansd[atom_type == self.atom_type_a]) / self.n_a_
        msd_b = np.sum(meansd[atom_type == self.atom_type_b]) / self.n_b_

        self.frame_ += 1
        return np.array([self.frame_, msd_a, msd_b, msd_t], dtype=np.float32)

    def unwrapped(self, atom_type, positions):
        """If the trajectory is unwrapped outside of the simulation box.

        Parameters
        ----------
        atom_type : list
            the type of the atoms

        positions : np.array
            the positions in the SoA convention (i.e. first all the x,
            then y and then z)

        Returns
        -------
        np.array
            with the frame in the first value, the msd of atom type a in the
            second, the msd of atom type b in the third and the total msd in
            the fourth.
        """
        msd_a, msd_b, msd_t = 0.0, 0.0, 0.0

        positions = np.split(positions, 3)
        meansd = np.zeros(self.natoms, dtype=np.float32)
        for i in range(3):
            xx = positions[i] - self.ref[i]
            meansd += xx * xx

        msd_t = np.sum(meansd) / self.natoms
        msd_a = np.sum(meansd[atom_type == self.atom_type_a]) / self.n_a_
        msd_b = np.sum(meansd[atom_type == self.atom_type_b]) / self.n_b_

        self.frame_ += 1
        return np.array([self.frame_, msd_a, msd_b, msd_t], dtype=np.float32)
