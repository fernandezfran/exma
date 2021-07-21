# -*- coding: utf-8 -*-

""" setup script """

import sysconfig
from setuptools import find_packages
from distutils.core import setup, Extension

""" long description of project """
with open('README.md') as file_readme:
    readme = file_readme.read()

""" requirements to setup exma """
with open('requirements.rst') as file_requirements:
    requirements = file_requirements.read().splitlines()

""" modules of exma that are written in C """
with open('exma_packages.rst') as file_exma_packages:
    exma_packages = file_exma_packages.read().splitlines()

CFLAGS = sysconfig.get_config_var('CFLAGS').split()
CFLAGS += ["-O3", "-ffast-math", "-fPIC", "-ftree-vectorize", "-march=native"]

C_modules = []
for mod in exma_packages:
    if mod == "exma": continue
    mod = mod.replace("exma.","")
    common = "exma/" + mod + "/" 

    library = common + "lib_" + mod
    src     = common + mod + ".c"
    depend  = common + mod + ".h"

    C_modules.append(Extension(library, sources=[src], depends=[depend],
                               extra_compile_args=CFLAGS))


""" setup """
setup(
    name='exma',
    version='0.2.0',
    description='extendable molecular dynamics analyzer',
    long_description=readme,
    long_description_content_type = 'text/markdown',
    packages=find_packages(include=exma_packages),
    author='Francisco Fernandez',
    author_email='fernandezfrancisco2195@gmail.com',
    url='https://github.com/fernandezfran/exma',
    license='MIT',
    install_requires=requirements,
    setup_requires=requirements,
    keywords=['exma', 'molecular-dynamics', 'data-analysis'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: C',
        'Topic :: Scientific/Engineering'
    ],
    ext_modules = C_modules
)
