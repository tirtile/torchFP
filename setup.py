# coding:utf-8

from setuptools import setup
# or
# from distutils.core import setup  

setup(
        name='torchfp',     # 包名字
        version='1.0',   # 包版本
        description='This is a parameter and FLOPs calculation tools for pytorch model',   # 简单描述
        author='tty',  # 作者
        author_email='2062898603@qq.com',  # 作者邮箱
        url='https://github.com/tirtile/torchFP/',      # 包的主页
        packages=['torchFP'],                 # 包
)
