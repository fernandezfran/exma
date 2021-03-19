# -*- coding: utf-8 -*-

""" setup script """

import sysconfig
from setuptools import find_packages
from distutils.core import setup, Extension

#### long description of project
with open('README.md') as file_readme:
    readme = file_readme.read()

#### requirements to install and setup exma
with open('requirements.rst') as file_requirements:
    requirements = file_requirements.read().splitlines()


#### modules of exma that are written in C
CFLAGS = sysconfig.get_config_var('CFLAGS').split()
CFLAGS += ["-O3", "-ffast-math", "-fPIC", "-ftree-vectorize", "-march=native"]

with open('exma_packages.rst') as file_exma_packages:
    exma_packages = file_exma_packages.read().splitlines()


boundary_mod = Extension('exma/boundary/lib_boundary',
                         sources=['exma/boundary/boundary.c'],
                         depends=['exma/boundary/boundary.h'],
                         extra_compile_args=CFLAGS)

cluster_mod = Extension('exma/cluster/lib_cluster',
                        sources=['exma/cluster/cluster.c'],
                        depends=['exma/cluster/cluster.h'],
                        extra_compile_args=CFLAGS)

CN_mod = Extension('exma/CN/lib_cn',
                   sources=['exma/CN/cn.c'],
                   depends=['exma/CN/cn.h'],
                   extra_compile_args=CFLAGS)

EN_mod = Extension('exma/EN/lib_en',
                   sources=['exma/EN/en.c'],
                   depends=['exma/EN/en.h'],
                   extra_compile_args=CFLAGS)

RDF_mod = Extension('exma/RDF/lib_rdf',
                    sources=['exma/RDF/rdf.c'],
                    depends=['exma/RDF/rdf.h'],
                    extra_compile_args=CFLAGS)

C_modules = [boundary_mod, cluster_mod, CN_mod, EN_mod, RDF_mod]


#### setup
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
    ext_modules = C_modules
)
