# -*- coding: utf-8 -*-

""" setup script """

from setuptools import find_packages, setup

with open('README.md') as file_readme:
    readme = file_readme.read()

requirements = ['numpy', 'pandas']

setup(
    name='emda',
    version='0.1.0',
    description='extendable molecular dynamics analyzer',
    long_description=readme,
    packages=find_packages(include=['emda']),
    author='Francisco Fernandez',
    author_email='fernandezfrancisco2195@gmail.com',
    url='https://github.com/fernandezfran/emda',
    license='MIT',
    install_requires=requirements,
    setup_requires=requirements,
    keywords=['emda', 'molecular-dynamics', 'data-analysis']
)
