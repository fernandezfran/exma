#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""An extendable molecular analyzer."""

# ============================================================================
# CONSTANTS
# ============================================================================

__author__ = """Francisco Fernandez"""
__email__ = "fernandezfrancisco2195@gmail.com"
__version__ = "0.5.5"


# ============================================================================
# IMPORTS
# ============================================================================

# core
from .core import AtomicSystem  # noqa

# distances
from .distances import pbc_distances  # noqa

# molecular dynamics observables
from .msd import MeanSquareDisplacement  # noqa
from .rdf import RadialDistributionFunction  # noqa
from .cn import CoordinationNumber  # noqa
from .cluster import EffectiveNeighbors, DBSCAN, sro  # noqa

# electrochemistry
from .electrochemistry import (  # noqa
    fractional_volume_change,  # noqa
    formation_energy,  # noqa
    voltage,  # noqa
)  # noqa
from .statistics import BlockAverage  # noqa

# io
from .io import *  # noqa
from .io.reader import read_xyz, read_lammpstrj, read_log_lammps  # noqa
from .io.writer import write_xyz, write_lammpstrj, write_in_lammps  # noqa
from .io.positions import Positions, spherical_nanoparticle, replicate  # noqa
