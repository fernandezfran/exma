#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of exma (https://github.com/fernandezfran/exma/).
# Copyright (c) 2021, Francisco Fernandez
# License: MIT
#   Full Text: https://github.com/fernandezfran/exma/blob/master/LICENSE

# =============================================================================
# DOCS
# =============================================================================

"""Installation of exma."""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import pathlib
import sysconfig

from setuptools import find_packages

from distutils.core import Extension, setup  # noqa: I100

# =============================================================================
# CONSTANTS
# =============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

REQUIREMENTS = [
    "matplotlib",
    "more-itertools",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
]

COMMON = "exma/lib/"
CFLAGS = sysconfig.get_config_var("CFLAGS").split()
CFLAGS += ["-O3", "-fPIC", "-ftree-vectorize", "-march=native"]
C_MODS = [
    Extension(
        COMMON + "lib_" + mod,
        sources=[COMMON + mod + ".c"],
        depends=[COMMON + mod + ".h"],
        extra_compile_args=CFLAGS,
    )
    for mod in ["cn", "pbc_distances", "rdf"]
]

with open(PATH / "exma" / "__init__.py") as fp:
    for line in fp.readlines():
        if line.startswith("__version__ = "):
            VERSION = line.split("=", 1)[-1].replace('"', "").strip()
            break

with open("README.md") as file_readme:
    LONG_DESCRIPTION = file_readme.read()


# =============================================================================
# FUNCTIONS
# =============================================================================

setup(
    name="exma",
    version=VERSION,
    description="extendable molecular dynamics analyzer",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    author="Francisco Fernandez",
    author_email="fernandezfrancisco2195@gmail.com",
    url="https://github.com/fernandezfran/exma",
    license="MIT",
    install_requires=REQUIREMENTS,
    setup_requires=REQUIREMENTS,
    keywords=["exma", "molecular-dynamics", "data-analysis"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: C",
        "Topic :: Scientific/Engineering",
    ],
    ext_modules=C_MODS,
)
