#!/usr/bin/env python

from setuptools import setup, find_namespace_packages


setup(
    name="project",
    version="0.0.0",
    description="Defect Detection based on Zero-Shot Instance Segmentation",
    author="leaf_ying,jeffchengtw,patrick0115",
    author_email="leaf.ying.work@gmail.com",
    url="https://github.com/guyleaf/Defect-Detection-ZSI",
    install_requires=["torch~=1.12.1", "torchvision~=0.13.0", "pytorch-lightning~=1.7.4"],
    packages=find_namespace_packages(),
)
