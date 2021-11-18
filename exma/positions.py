#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""The exma.positions module include a class to define atom positions."""

# =============================================================================
# IMPORTS
# =============================================================================
import itertools as it

import numpy as np


class Positions:
    """Define the positions of the atoms in an orthogonal lattice.

    Note that the density is defined by the input parameters as follow:
    rho = natoms / (box_size^3)

    Parameters
    ----------
    natoms : int
        number of atoms

    box_size : float
        box size in each direction (cubic)
    """

    def __init__(self, natoms, box_size):

        self.natoms = natoms
        self.box_size = box_size

    def sc(self):
        """Simple cubic.

        This cell is characterized by having an atom in each of its vertices.

        Returns
        -------
        dict :
            with the keys `natoms`, `box`, `x`, `y`, `z`, the number of
            atoms, the box size and the xyz of the atoms, respectively.
        """
        nside = np.cbrt(self.natoms, dtype=np.float32)
        tmp = np.intc(nside)
        if nside % tmp != 0:
            raise ValueError("Number of atoms must be a power of three")

        s_range = range(int(nside))
        positions = list(it.product(s_range, repeat=3))

        positions = np.array(positions, dtype=np.float32)
        positions = np.transpose(positions) * (self.box_size / nside)

        return {
            "natoms": self.natoms,
            "box": np.full(3, self.box_size, dtype=np.float32),
            "x": positions[0],
            "y": positions[1],
            "z": positions[2],
        }

    def bcc(self):
        """Body-centered cubic.

        This cell is characterized by having one atom in each of its vertices
        and an additional atom in the center of the cube.

        Returns
        -------
        dict :
            with the keys `natoms`, `box`, `x`, `y`, `z`, the number of
            atoms, the box size and the xyz of the atoms, respectively.
        """
        nside = np.cbrt(self.natoms / 2, dtype=np.float32)
        tmp = np.intc(nside)
        if nside % tmp != 0:
            raise ValueError(
                "Number of atoms must be a power of three multiplied by two"
            )

        s_range = range(int(nside))
        p0 = list(it.product(s_range, repeat=3))

        # bcc lattice vectors: (0, 0, 0) and (0.5, 0.5, 0.5)
        p0 = np.array(p0)
        p1 = p0 + np.full((len(p0), 3), [0.5, 0.5, 0.5])

        positions = np.concatenate((p0, p1), dtype=np.float32)
        positions = np.transpose(positions) * (self.box_size / nside)

        return {
            "natoms": self.natoms,
            "box": np.full(3, self.box_size, dtype=np.float32),
            "x": positions[0],
            "y": positions[1],
            "z": positions[2],
        }

    def fcc(self):
        """Face-centered cubic.

        This cell is characterized by having one atom in each of its vertices
        and an additional atom in each of its faces.

        Returns
        -------
        dict :
            with the keys `natoms`, `box`, `x`, `y`, `z`, the number of
            atoms, the box size and the xyz of the atoms, respectively.
        """
        nside = np.cbrt(self.natoms / 4, dtype=np.float32)
        tmp = np.intc(nside)
        if nside % tmp != 0:
            raise ValueError(
                "Number of atoms must be a power of three multiplied by four"
            )

        s_range = range(int(nside))
        p0 = list(it.product(s_range, repeat=3))

        # fcc lattice vectors:
        # (0, 0, 0) (0.5, 0.5, 0) (0.5, 0, 0.5) (0, 0.5, 0.5)
        p0 = np.array(p0)
        p1 = p0 + np.full((len(p0), 3), [0.5, 0.5, 0.0])
        p2 = p0 + np.full((len(p0), 3), [0.5, 0.0, 0.5])
        p3 = p0 + np.full((len(p0), 3), [0.0, 0.5, 0.5])

        positions = np.concatenate((p0, p1, p2, p3), dtype=np.float32)
        positions = np.transpose(positions) * (self.box_size / nside)

        return {
            "natoms": self.natoms,
            "box": np.full(3, self.box_size, dtype=np.float32),
            "x": positions[0],
            "y": positions[1],
            "z": positions[2],
        }

    def dc(self):
        """Diamond cubic.

        The typical structure of a diamond, defined by eight sites.

        Returns
        -------
        dict :
            with the keys `natoms`, `box`, `x`, `y`, `z`, the number of
            atoms, the box size and the xyz of the atoms, respectively.
        """
        nside = np.cbrt(self.natoms / 8, dtype=np.float32)
        tmp = np.intc(nside)
        if nside % tmp != 0:
            raise ValueError("Number of atoms not valid")

        s_range = range(int(nside))
        p0 = list(it.product(s_range, repeat=3))

        p0 = np.array(p0)
        p1 = p0 + np.full((len(p0), 3), [0.25, 0.75, 0.25])
        p2 = p0 + np.full((len(p0), 3), [0.00, 0.00, 0.50])
        p3 = p0 + np.full((len(p0), 3), [0.25, 0.25, 0.75])
        p4 = p0 + np.full((len(p0), 3), [0.00, 0.50, 0.00])
        p5 = p0 + np.full((len(p0), 3), [0.75, 0.75, 0.75])
        p6 = p0 + np.full((len(p0), 3), [0.50, 0.00, 0.00])
        p7 = p0 + np.full((len(p0), 3), [0.75, 0.25, 0.25])
        p8 = p0 + np.full((len(p0), 3), [0.50, 0.50, 0.50])

        positions = np.concatenate(
            (p1, p2, p3, p4, p5, p6, p7, p8), dtype=np.float32
        )
        positions = np.transpose(positions) * (self.box_size / nside)

        return {
            "natoms": self.natoms,
            "box": np.full(3, self.box_size, dtype=np.float32),
            "x": positions[0],
            "y": positions[1],
            "z": positions[2],
        }

    def spherical_nanoparticle(self, box_size, positions, rcut):
        """Cut a defined structure to give a spherical nanoparticle.

        If rcut is greater than half the cell length, the cell will be
        replicated to give a nanoparticle of the desired radius.

        Parameters
        ----------
        box_size : np.array
            box size in each direction x, y, z.

        positions : np.array
            the positions of the atoms in a lattice that wants to be
            replicated. It must first have all the x's of the atoms, then the
            y's and then the z's concatenated.

        rcut : float
            the radius of the nanoparticle.

        Returns
        -------
        dict :
            with the keys `natoms`, `x`, `y`, `z`, the number of atoms and
            the xyz of the atoms, respectively.
        """
        self.box_size = box_size
        x, y, z = np.split(positions, 3)

        n = np.intc(np.ceil(rcut / np.max(self.box_size)))
        boxes = list(it.product(range(-n, n + 1), repeat=3))

        npx, npy, npz = [], [], []
        for box in boxes:
            for i in range(len(x)):
                xx = self.box_size[0] * (x[i] + box[0])
                yy = self.box_size[1] * (y[i] + box[1])
                zz = self.box_size[2] * (z[i] + box[2])

                if np.linalg.norm([xx, yy, zz]) <= rcut:
                    npx.append(xx)
                    npy.append(yy)
                    npz.append(zz)

        return {
            "natoms": len(npx),
            "x": np.asarray(npx, dtype=np.float32),
            "y": np.asarray(npy, dtype=np.float32),
            "z": np.asarray(npz, dtype=np.float32),
        }

    def replicate(self, box_size, atom_type, positions, nrf):
        """Replicate a crystalline system in each direction.

        Parameters
        ----------
        box_size : np.array
            box size in each direction x, y, z.

        atom_type : list of integers
            the type of the atoms.

        positions : np.array
            the positions of the atoms in a lattice that wants to be
            replicated. It must first have all the x's of the atoms, then the
            y's and then the z's concatenated.

        nrf : list of integers.
            the integers must be greater than or equal to 1 and indicate the
            replication factor in each x, y, z direction, respectively. Value
            equal to 1 means that only the current cell in that direction is
            considered.

        Returns
        -------
        dict :
            with the keys `natoms`, `box`, types`, x`, `y`, `z`, the number of
            atoms, the box size, the types of the atoms and the xyz of the
            atoms, respectively.
        """
        self.atom_type = atom_type
        self.box_size = box_size

        x, y, z = np.split(positions, 3)
        boxes = list(it.product(range(np.max(nrf)), repeat=3))
        newx, newy, newz = [], [], []
        for box in boxes:
            if (box[0] >= nrf[0]) or (box[1] >= nrf[1]) or (box[2] >= nrf[2]):
                continue

            for i in range(self.natoms):
                newx.append(self.box_size[0] * (x[i] + box[0]))
                newy.append(self.box_size[1] * (y[i] + box[1]))
                newz.append(self.box_size[2] * (z[i] + box[2]))

        self.natoms = len(newx)
        self.box_size = np.array(
            [
                nrf[0] * self.box_size[0],
                nrf[1] * self.box_size[1],
                nrf[2] * self.box_size[2],
            ]
        )
        self.atom_type = self.atom_type * np.prod(nrf)

        return {
            "natoms": self.natoms,
            "box": self.box_size,
            "types": self.atom_type,
            "x": np.asarray(newx, dtype=np.float32),
            "y": np.asarray(newy, dtype=np.float32),
            "z": np.asarray(newz, dtype=np.float32),
        }
