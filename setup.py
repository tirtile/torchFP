#!/usr/bin/env python
import os
import shutil
import sys
from setuptools import setup, find_packages


readme = open('README.md').read()

VERSION = '1.0.0'

requirements = [
    'torch',
]

setup(
    # Metadata
    name='torchfp',
    version=VERSION,
    author='tty',
    author_email='',
    url='https:/github.com/tirtile/torchFP/',
    description='A tool to count the FLOPs of PyTorch model.',
    long_description=readme,
    license='MIT',

    # Package info
    packages=find_packages(exclude=('*test*',)),

    #
    zip_safe=True,
    install_requires=requirements,

    # Classifiers
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
