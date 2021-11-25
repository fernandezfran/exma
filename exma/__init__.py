#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# ======================================================================
# DOCS
# ======================================================================

"""An extendable molecular analyzer."""

# ======================================================================
# CONSTANTS
# ======================================================================

__author__ = """Francisco Fernandez"""
__email__ = "fernandezfrancisco2195@gmail.com"
__version__ = "0.3.0"


# ======================================================================
# IMPORTS
# ======================================================================

from .io.reader import XYZ, LAMMPS, read_log_lammps  # noqa
from .io.writer import XYZ, LAMMPS, in_lammps  # noqa
from .msd import monoatomic, diatomic  # noqa
from .rdf import monoatomic, diatomic  # noqa
from .cn import monoatomic, diatomic  # noqa
from .cluster import DBSCAN, EffectiveNeighbors, sro  # noqa
from .electrochemistry import Electrochemistry  # noqa
from .positions import Positions, spherical_nanoparticle, replicate  # noqa
from .statistics import block_average  # noqa
