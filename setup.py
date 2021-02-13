from setuptools import find_packages, setup

setup(
    name='emda',
    packages=find_packages(include=['emda']),
    version='0.1.0',
    description='extendable molecular dynamics analyzer',
    author='Francisco Fernandez',
    license='MIT',
    install_requires=['numpy', 'pandas'],
    setup_requires=['numpy', 'pandas'],
)
