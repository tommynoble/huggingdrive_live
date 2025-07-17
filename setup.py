#!/usr/bin/env python3
"""
Setup script for HuggingDrive
"""

from setuptools import setup, find_packages
import os


# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [
            line.strip() for line in fh if line.strip() and not line.startswith("#")
        ]


setup(
    name="huggingdrive",
    version="1.0.0",
    author="HuggingDrive Team",
    author_email="",
    description="A tool for managing Hugging Face models on external drives",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "huggingdrive=huggingdrive_cli:main",
            "huggingdrive-gui=huggingdrive:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="huggingface transformers models external-drive ai machine-learning",
    project_urls={
        "Bug Reports": "",
        "Source": "",
        "Documentation": "",
    },
)
