#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Implementation of functions of interest in electrochemistry."""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pandas as pd

import scipy.interpolate

# ============================================================================
# CLASSES
# ============================================================================


class Electrochemistry:
    """Electrochemistry functionalities.

    Parameters
    ----------
    df : pd.DataFrame
        a pandas dataframe with the values of `x` (concentrations) in
        the first column and then columns with different thermodynamic
        equilibrium values, optionally with their respective errors,
        corresponding to each `x` value.

    Notes
    -----
    The names of the columns in the df must be included in the following
    names:

    `x`: `x` values

    `PotEng`: potential energy

    `KinEng`: kinetic energy

    `TotEng`: total energy

    `Press`: pressure

    `Vol`: volume

    `Temp`: temperature

    `N`: total number of atoms

    `NA`: number of atoms of type A

    `NB`: number of atoms of type B

    ...

    Columns with error have the keyword `err` on, i.e. `errVol` for
    `Vol`, for example.

    """

    def __init__(self, df):
        self.df = df

        self.dffe_ = None

    def fractional_volume_change(self, reference_atoms, reference_volume):
        r"""Fractional volume change (fvc) with respect to A.

        The *fvc* points are calculated using a normalization relative to
        the number of atoms in the reference structure and its volume

        .. math::
            fvc = {\frac{N_{ref}}{V_{ref}}} \left(
                {\frac{V_x}{N_x}} - {\frac{V_{ref}}{N_{ref}}}
            \right)

        where :math:`N_{ref}` and :math:`V_{ref}` are the number of atoms
        of type A and the volume of the reference structure, :math:`N_x` and
        :math:`V_x` are the number of atoms of type A and the volume of the
        structure at the :math:`x` concentration.

        Parameters
        ----------
        reference_atoms : int
            the number of atoms in the reference structure.

        reference_volume : float
            the volume, in the corresponding units, of the reference structue.

        Returns
        -------
        pd.DataFrame
            The fractional volume change corresponding values to each `x`
            values and the respective error if it was possible to calculate.

        Raises
        ------
        KeyError
            if the number of atoms of type A or the volume are not defined.

        Notes
        -----
        The volume, and the number of atoms of type A in the structure must be
        defined in the corresponding df.
        """
        if ("NA" not in self.df) or ("Vol" not in self.df):
            raise KeyError(
                "The number of atoms of type A or the volume of the "
                "structures are not specified in the input DataFrame"
            )

        fvc = [
            (reference_atoms / reference_volume)
            * ((volume / natoms) - (reference_volume / reference_atoms))
            for natoms, volume in zip(self.df.NA, self.df.Vol)
        ]

        self.dffvc_ = {"x": self.df.x, "fvc": np.array(fvc, dtype=np.float32)}

        if "errVol" in self.df:
            errfvc = [
                (reference_atoms / reference_volume) * (errvolume / natoms)
                for natoms, errvolume in zip(self.df.NA, self.df.errVol)
            ]
            self.dffvc_["errfvc"] = errfvc = np.array(errfvc, dtype=np.float32)

        self.dffvc_ = pd.DataFrame(self.dffvc_)

        return self.dffvc_

    def formation_energy(self, reference_energy_a, reference_energy_b):
        r"""Ideal approximation to the formation energy (fe).

        .. math::
            E_f(x) = E_x - (x \cdot E_b + E_a)

        where :math:`E_x` is the energy per atom of type A, :math:`x` the
        concentration and :math:`E_a` and :math:`E_b` the cohesive energies
        of A and B bulk materials.

        Parameters
        ----------
        reference_energy_a : float
            the pure energy of element A in bulk

        reference_energy_b : float
            the pure energy of element B in bulk

        Returns
        -------
        pd.DataFrame
            The formation energy corresponding values to each x values
            and the respective error if it was possible to calculate.

        Raises
        ------
        KeyError
            if the `x` values or the potential energies are not defined.

        Notes
        -----
        The number of atoms of A is fixed and B is the varying element.
        """
        if (
            ("x" not in self.df)
            or ("NA" not in self.df)
            or ("PotEng" not in self.df)
        ):
            raise KeyError(
                "The x values, the number of atoms of type A or the "
                "corresponding potential energies are not specified in "
                "the input DataFrame"
            )

        fe = [
            energy / na - (x * reference_energy_b + reference_energy_a)
            for x, na, energy in zip(self.df.x, self.df.NA, self.df.PotEng)
        ]
        self.dffe_ = {"x": self.df.x, "fe": np.array(fe, dtype=np.float32)}

        if "errPotEng" in self.df:
            self.dffe_["errfe"] = np.asarray(
                self.df.errPotEng, dtype=np.float32
            )

        self.dffe_ = pd.DataFrame(self.dffe_)
        return self.dffe_

    def voltage(self, **kwargs):
        r"""Approximation to the voltage curve.

        The formation energies can be used as an approximation to the Gibbs
        formation energy, then the potential :math:`V` is given by:

        .. math::
            V(x) = -{\frac{dE_f(x)}{dx}}

        Parameters
        ----------
        **kwargs
            additional keyword arguments that are passed and are documented
            in `scipy.interpolate.UnivariateSpline`

        Returns
        -------
        pd.DataFrame
            The value of the formation energy after the spline and the
            estimated value to the voltage for each x value.

        Raises
        ------
        NameError
            If formation_energy was not already called

        Notes
        -----
        The voltage is approximated from the formation energies values, so you
        first need to run Electrochemistry.formation_energy(...)
        """
        if self.dffe_ is None:
            raise NameError("The formation energies were not calculated yet.")

        weights = self.dffe_.errfe if "errfe" in self.dffe_ else None
        spline = scipy.interpolate.UnivariateSpline(
            self.dffe_.x, self.dffe_.fe, w=weights, **kwargs
        )
        dspline = spline.derivative()

        fe_spline = [spline(x) for x in self.dffe_.x]
        voltage = [-dspline(x) for x in self.dffe_.x]

        return pd.DataFrame(
            {
                "x": self.df.x,
                "fe_spline": np.array(fe_spline, dtype=np.float32),
                "voltage": np.array(voltage, dtype=np.float32),
            }
        )
