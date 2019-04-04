#!/usr/bin/env python
# coding=utf-8

from setuptools import setup

'''
把redis服务打包成C:\Python27\Scripts下的exe文件
'''

setup(
    name="torchfp",
    version="1.0",
    author="tirtile",
    author_email='2062898603@qq.com',
    description=("FLOPs and parameters calculation tool for pytorch model"),
    license="MIT",
    url="https://github.com/tirtile/torchFP/",

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
