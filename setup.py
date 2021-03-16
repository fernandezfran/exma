# -*- coding: utf-8 -*-

""" setup script """

from setuptools import find_packages#, setup
from distutils.core import setup, Extension
import sysconfig

with open('README.md') as file_readme:
    readme = file_readme.read()

requirements = ['sklearn', 'more-itertools', 'numpy', 'pandas']

exma_packages = ['exma', 'exma.RDF']

CFLAGS = sysconfig.get_config_var('CFLAGS').split()
CFLAGS += ["-O3", "-ffast-math", "-funroll-loops", "-march=native",
           "-ftree-vectorize"]

RDF_mod = Extension('exma/RDF/lib_rdf',
                    sources=['exma/RDF/rdf.c'],
                    depends=['exma/RDF/rdf.h'],
                    extra_compile_args=CFLAGS)


setup(
    name='exma',
    version='0.1.0',
    description='extendable molecular dynamics analyzer',
    long_description=readme,
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
        'Programming Language :: C'
        'Topic :: Scientific/Engineering'
    ],
    ext_modules = [RDF_mod]
)
