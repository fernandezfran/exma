#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Includes class and functions to define atom positions.

It is primarily developed to generate initial structures returned in the
form of a `exma.core.AtomicSystem` for writing to output files with the writer
classes/function that serve as initial conditions for simulations.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import itertools as it

import numpy as np

from ..core import AtomicSystem

# ============================================================================
# CLASSES
# ============================================================================


class Positions:
    r"""Define the positions of the atoms in an orthogonal lattice.

    Parameters
    ----------
    natoms : int
        number of atoms

    box_size : float
        box size in each direction (cubic)

    Notes
    -----
    Note that the density is defined by the input parameters as follow:

    .. math::
        {\rho} = {\frac{N}{L^3}}

    where `N` is the number of atoms and `L` the box lenght in each
    direction.
    """

    def __init__(self, natoms, box_size):

        self.natoms = natoms
        self.box_size = box_size

    def sc(self):
        """Simple-cubic crystal.

        This cell is characterized by having an atom in each of its vertices.

        Returns
        -------
        `exma.core.AtomicSystem`
            This have all the information of the configurations of the system.

        Raises
        ------
        ValueError
            If the number of atoms does not correspond with the number of
            sites that a sc crystal structure has.
        """
        nside = np.cbrt(self.natoms, dtype=np.float32)
        nboxes = np.intc(nside)
        if nside % nboxes != 0:
            raise ValueError("Number of atoms must be a power of three")

        s_range = range(nboxes)
        positions = it.product(s_range, repeat=3)

        positions = np.array(list(positions), dtype=np.float32)
        positions = np.transpose(positions) * (self.box_size / nside)

        return AtomicSystem(
            natoms=self.natoms,
            box=np.full(3, self.box_size, dtype=np.float32),
            x=positions[0],
            y=positions[1],
            z=positions[2],
        )

    def bcc(self):
        """Body-centered cubic crystal.

        This cell is characterized by having one atom in each of its vertices
        and an additional atom in the center of the cube.

        Returns
        -------
        `exma.core.AtomicSystem`
            This have all the information of the configurations of the system.

        Raises
        ------
        ValueError
            If the number of atoms does not correspond with the number of
            sites that a bcc crystal structure has.
        """
        nside = np.cbrt(self.natoms / 2, dtype=np.float32)
        nboxes = np.intc(nside)
        if nside % nboxes != 0:
            raise ValueError(
                "Number of atoms must be a power of three multiplied by two"
            )

        s_range = range(nboxes)
        p0 = it.product(s_range, repeat=3)

        # bcc lattice vectors: (0, 0, 0) and (0.5, 0.5, 0.5)
        p0 = np.array(list(p0))
        p1 = p0 + np.full((np.power(nboxes, 3), 3), [0.5, 0.5, 0.5])

        positions = np.concatenate((p0, p1), dtype=np.float32)
        positions = np.transpose(positions) * (self.box_size / nside)

        return AtomicSystem(
            natoms=self.natoms,
            box=np.full(3, self.box_size, dtype=np.float32),
            x=positions[0],
            y=positions[1],
            z=positions[2],
        )

    def fcc(self):
        """Face-centered cubic crystal.

        This cell is characterized by having one atom in each of its vertices
        and an additional atom in each of its faces.

        Returns
        -------
        `exma.core.AtomicSystem`
            This have all the information of the configurations of the system.

        Raises
        ------
        ValueError
            If the number of atoms does not correspond with the number of
            sites that a fcc crystal structure has.
        """
        nside = np.cbrt(self.natoms / 4, dtype=np.float32)
        nboxes = np.intc(nside)
        if nside % nboxes != 0:
            raise ValueError(
                "Number of atoms must be a power of three multiplied by four"
            )

        s_range = range(nboxes)
        p0 = it.product(s_range, repeat=3)

        # fcc lattice vectors:
        # (0, 0, 0) (0.5, 0.5, 0) (0.5, 0, 0.5) (0, 0.5, 0.5)
        p0 = np.array(list(p0))
        p1 = p0 + np.full((np.power(nboxes, 3), 3), [0.5, 0.5, 0.0])
        p2 = p0 + np.full((np.power(nboxes, 3), 3), [0.5, 0.0, 0.5])
        p3 = p0 + np.full((np.power(nboxes, 3), 3), [0.0, 0.5, 0.5])

        positions = np.concatenate((p0, p1, p2, p3), dtype=np.float32)
        positions = np.transpose(positions) * (self.box_size / nside)

        return AtomicSystem(
            natoms=self.natoms,
            box=np.full(3, self.box_size, dtype=np.float32),
            x=positions[0],
            y=positions[1],
            z=positions[2],
        )

    def dc(self):
        """Diamond cubic crystal.

        The typical structure of a diamond, defined by eight sites.

        Returns
        -------
        `exma.core.AtomicSystem`
            This have all the information of the configurations of the system.

        Raises
        ------
        ValueError
            If the number of atoms does not correspond with the number of
            sites that a dc crystal structure has.
        """
        nside = np.cbrt(self.natoms / 8, dtype=np.float32)
        nboxes = np.intc(nside)
        if nside % nboxes != 0:
            raise ValueError("Number of atoms not valid")

        s_range = range(nboxes)
        p0 = it.product(s_range, repeat=3)

        p0 = np.array(list(p0))
        p1 = p0 + np.full((np.power(nboxes, 3), 3), [0.25, 0.75, 0.25])
        p2 = p0 + np.full((np.power(nboxes, 3), 3), [0.00, 0.00, 0.50])
        p3 = p0 + np.full((np.power(nboxes, 3), 3), [0.25, 0.25, 0.75])
        p4 = p0 + np.full((np.power(nboxes, 3), 3), [0.00, 0.50, 0.00])
        p5 = p0 + np.full((np.power(nboxes, 3), 3), [0.75, 0.75, 0.75])
        p6 = p0 + np.full((np.power(nboxes, 3), 3), [0.50, 0.00, 0.00])
        p7 = p0 + np.full((np.power(nboxes, 3), 3), [0.75, 0.25, 0.25])
        p8 = p0 + np.full((np.power(nboxes, 3), 3), [0.50, 0.50, 0.50])

        positions = np.concatenate(
            (p1, p2, p3, p4, p5, p6, p7, p8), dtype=np.float32
        )
        positions = np.transpose(positions) * (self.box_size / nside)

        return AtomicSystem(
            natoms=self.natoms,
            box=np.full(3, self.box_size, dtype=np.float32),
            x=positions[0],
            y=positions[1],
            z=positions[2],
        )


# ============================================================================
# FUNCTIONS
# ============================================================================


def spherical_nanoparticle(frame, rcut):
    """Cut a defined structure to give a spherical nanoparticle.

    Parameters
    ----------
    frame : `exma.core.AtomicSystem`
        This have all the information of the configurations of the system.

    rcut : float
        the radius of the nanoparticle.

    Returns
    -------
    `exma.core.AtomicSystem`
        This have all the information of the configurations of the system.

    Notes
    -----
    If `rcut` is greater than half the cell length, the cell will be
    replicated to give a nanoparticle of the desired radius.
    """
    box_size = frame.box
    x = frame.x
    y = frame.y
    z = frame.z

    # scale the positions if they are not in fractions of the box size
    if np.max(frame.x) > 0.5 * box_size[0]:
        x = x / box_size[0]
        y = y / box_size[1]
        z = z / box_size[2]

    n = np.intc(np.ceil(rcut / np.min(box_size)))
    boxes = it.product(range(-n, n + 1), repeat=3)

    npty, npx, npy, npz = [], [], [], []
    for box in boxes:
        for i in range(len(x)):
            xx = box_size[0] * (x[i] + box[0])
            yy = box_size[1] * (y[i] + box[1])
            zz = box_size[2] * (z[i] + box[2])

            if np.linalg.norm([xx, yy, zz]) <= rcut:
                npty.append(frame.types[i])

                npx.append(xx)
                npy.append(yy)
                npz.append(zz)

    return AtomicSystem(
        natoms=len(npx),
        types=npty,
        x=np.asarray(npx, dtype=np.float32),
        y=np.asarray(npy, dtype=np.float32),
        z=np.asarray(npz, dtype=np.float32),
    )


def replicate(frame, nrf):
    """Replicate a crystalline system in each direction.

    Parameters
    ----------
    frame : `exma.core.AtomicSystem`
        This have all the information of the configurations of the system.

    nrf : list
        three integers that must be greater than or equal to 1 and
        indicates the replication factor in each x, y, z direction,
        respectively. Value equal to 1 means that only the current cell
        in that direction is considered.

    Returns
    -------
    `exma.core.AtomicSystem`
        This have all the information of the configurations of the system.
    """
    box_size = frame.box
    x = frame.x
    y = frame.y
    z = frame.z

    # scale the positions if they are not in fractions of the box size
    if np.max(frame.x) > 0.5 * box_size[0]:
        x = x / box_size[0]
        y = y / box_size[1]
        z = z / box_size[2]

    boxes = it.product(range(np.max(nrf)), repeat=3)
    newx, newy, newz = [], [], []
    for box in boxes:
        if (box[0] >= nrf[0]) or (box[1] >= nrf[1]) or (box[2] >= nrf[2]):
            continue

        for i in range(frame.natoms):
            newx.append(box_size[0] * (x[i] + box[0]))
            newy.append(box_size[1] * (y[i] + box[1]))
            newz.append(box_size[2] * (z[i] + box[2]))

    natoms = len(newx)
    box_size = np.array(
        [nrf[0] * box_size[0], nrf[1] * box_size[1], nrf[2] * box_size[2]]
    )
    atom_type = np.tile(frame.types, np.prod(nrf))

    return AtomicSystem(
        natoms=natoms,
        box=box_size,
        types=atom_type,
        x=np.asarray(newx, dtype=np.float32),
        y=np.asarray(newy, dtype=np.float32),
        z=np.asarray(newz, dtype=np.float32),
    )
