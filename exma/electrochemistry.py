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

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np

import pandas as pd

import scipy.interpolate

# ============================================================================
# FUNCTIONS
# ============================================================================


def fractional_volume_change(df, reference_atoms, reference_volume):
    r"""Fractional volume change (fvc) with respect to an element A.

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
    df : pd.DataFrame
        a `pd.DataFrame` with the values of `x` (concentrations) in
        the first column and then columns with the number of atoms of type A
        in each structure, the corresponding equilibrium values of the volume
        and, optionally, with its respective error.

    reference_atoms : int
        the number of atoms in the reference structure.

    reference_volume : float
        the volume, in the corresponding units, of the reference structue.

    Returns
    -------
    pd.DataFrame
        The input `pd.DataFrame` with extra columns with the fractional volume
        change and the respective error if it was possible to calculate, `fvc`
        and `err_fvc`, respectively.

    Raises
    ------
    KeyError
        if the number of atoms of type A or the volume are not defined in
        the `pd.DataFrame` as `"natoms_a"` and `"volume"`, respectively.

    Notes
    -----
    The names of the columns in the df must be the following ones:

    `"x"`: `x` values

    `"volume"`: volume

    `"natoms_a"`: number of atoms of type A

    `"err_volume"`: the error of each `"volume"` point, optional.
    """
    if ("natoms_a" not in df) or ("volume" not in df):
        raise KeyError(
            "The number of atoms of type A or the volume of the "
            "structures are not specified in the input DataFrame"
        )

    sfc = reference_volume / reference_atoms
    fvc = [
        ((volume / natoms) - sfc) / sfc
        for natoms, volume in zip(df.natoms_a, df.volume)
    ]

    df["fvc"] = np.array(fvc, dtype=np.float32)

    if "err_volume" in df:
        errfvc = df.err_volume / (df.natoms_a * sfc)
        df["err_fvc"] = np.array(errfvc, dtype=np.float32)

    return df


def formation_energy(df, reference_energy_a, reference_energy_b, fetype="x"):
    r"""Ideal approximation to the formation energy (FE).

    .. math::
        E_f(x) = E_x - (x \cdot E_b + E_a)

    where :math:`E_x` is the energy per atom of type A, :math:`x` the
    concentration and :math:`E_a` and :math:`E_b` the cohesive energies
    of A and B bulk materials.

    The formation energy relative to the end members can also be calculated

    .. math::
        E_f(x) = \frac{E - n_a \cdot E_a - n_b \cdot E_b}{n_a + n_b}

    Parameters
    ----------
    df : pd.DataFrame
        a `pd.DataFrame` with the values of `x` (concentrations) in
        the first column and then columns with the number of atoms of types
        A and the equilibrium values of the potential energy and, optionally,
        with its respective error.

    reference_energy_a : float
        the pure energy of element A in bulk

    reference_energy_b : float
        the pure energy of element B in bulk

    fetype : str, default="x"
        `"x"` or `"relative"`, where the first one is the formation energy as
        a function of concentrations and the second one is relative to end
        members

    Returns
    -------
    pd.DataFrame
        The input `pd.DataFrame` with the formation energy corresponding
        values, `"fe"`, and the respective error if it was possible to
        calculate, `"err_fe"`.

    Notes
    -----
    The names of the columns in the df must include:

    `"x"` o `"mol"`: `x` or `mol` fraction values

    `"natoms_a"`: number of atoms of type A

    `"natoms_b"`: number of atoms of type B

    `"epot"`: potential energy

    `"err_epot"`: the error of each `"epot"` point, optional.
    """
    if fetype == "x":
        if ("x" not in df) or ("natoms_a" not in df) or ("epot" not in df):
            raise KeyError(
                "The x values, the number of atoms of type A or the "
                "corresponding potential energies are not specified in "
                "the input DataFrame"
            )

        fe = [
            epot / na - (x * reference_energy_b + reference_energy_a)
            for x, na, epot in zip(df.x, df.natoms_a, df.epot)
        ]

    elif fetype == "relative":
        if (
            ("natoms_a" not in df)
            or ("natoms_b" not in df)
            or ("epot" not in df)
        ):
            raise KeyError(
                "The number of atoms of type A or B or the corresponding "
                "potential energies are not specified in the input DataFrame"
            )

        fe = [
            (epot - na * reference_energy_a - nb * reference_energy_b)
            / (na + nb)
            for na, nb, epot in zip(df.natoms_a, df.natoms_b, df.epot)
        ]

    df["fe"] = np.array(fe, dtype=np.float32)

    if "err_epot" in df:
        if fetype == "x":
            natoms = df.natoms_a
        elif fetype == "relative":
            natoms = df.natoms_a + df.natoms_b

        df["err_fe"] = np.asarray(df.err_epot / natoms, dtype=np.float32)

    return df


def voltage(df, nums=50, **kwargs):
    r"""Approximation to the voltage curve.

    The formation energies can be used as an approximation to the Gibbs
    formation energy, then the potential :math:`V` is given by:

    .. math::
        V(x) = -{\frac{dE_f(x)}{dx}}

    Parameters
    ----------
    df : pd.DataFrame
        a `pd.DataFrame` with the values of `x` (concentrations), the formation
        energy, `"fe"`, and, optionally, the respective error, `"err_fe"`.

    nums : int, default=50
        number of points at which to evaluate the spline and its
        derivative

    **kwargs
        additional keyword arguments that are passed and are documented
        in `scipy.interpolate.UnivariateSpline`

    Returns
    -------
    pd.DataFrame
        A `pd.DataFrame` with the value of the formation energy after
        the spline and the estimated value to the voltage in function of
        `x`.
    """
    weights = df.errfe if "err_fe" in df else None
    spline = scipy.interpolate.UnivariateSpline(
        df.x, df.fe, w=weights, **kwargs
    )
    dspline = spline.derivative()

    xpoints = np.linspace(np.min(df.x), np.max(df.x), nums)
    fe_spline = [spline(x) for x in xpoints]
    voltage = [-dspline(x) for x in xpoints]

    return pd.DataFrame(
        {
            "x": xpoints,
            "fe_spline": np.array(fe_spline, dtype=np.float32),
            "voltage": np.array(voltage, dtype=np.float32),
        }
    )
