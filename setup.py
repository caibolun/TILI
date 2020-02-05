#!/usr/bin/env python
# coding=utf-8
'''
@Author: ArlenCai
@Date: 2020-02-04 14:14:12
@LastEditTime : 2020-02-04 18:27:31
'''
from distutils.core import setup, Extension

tili = Extension(
    "tili",
)

setup(
    name="TILI",
    version="0.0.1",
    description="TILI: Turbo Image Loading Library for Pytorch",
    author="Arlen Cai",
    author_email="arlencai@tencent.com",
    ext_modules=[tili],
)