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

from exma.positions import Positions, spherical_nanoparticle, replicate  # noqa
from exma.cluster import DBSCAN  # noqa
from exma.cn import monoatomic, diatomic  # noqa
from exma.electrochemistry import Electrochemistry  # noqa
from exma.en import EffectiveNeighbors  # noqa
from exma.msd import monoatomic, diatomic  # noqa
from exma.rdf import monoatomic, diatomic  # noqa
from exma.reader import xyz, lammpstrj  # noqa
from exma.sro import sro  # noqa
from exma.statistics import block_average  # noqa
from exma.writer import xyz, lammpstrj, in_lammps  # noqa
