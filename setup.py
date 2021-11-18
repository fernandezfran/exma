# -*- coding: utf-8 -*-

"""Installation of exma."""

import sysconfig
from setuptools import find_packages
from distutils.core import Extension, setup

with open("README.md") as file_readme:
    readme = file_readme.read()

REQUIREMENTS = ["scikit-learn", "scipy", "more-itertools", "numpy"]

CFLAGS = sysconfig.get_config_var("CFLAGS").split()
CFLAGS += ["-O3", "-ffast-math", "-fPIC", "-ftree-vectorize", "-march=native"]

C_modules = []
EXMA_PACKAGES = ["exma.cluster", "exma.cn", "exma.en", "exma.rdf"]
for mod in EXMA_PACKAGES:
    mod = mod.replace("exma.", "")
    common = "exma/" + mod + "/"

    library = common + "lib_" + mod
    src = common + mod + ".c"
    depend = common + mod + ".h"

    C_modules.append(
        Extension(
            library, sources=[src], depends=[depend], extra_compile_args=CFLAGS
        )
    )


setup(
    name="exma",
    version="0.2.0",
    description="extendable molecular dynamics analyzer",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["exma"] + EXMA_PACKAGES),
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
    ext_modules=C_modules,
)
