"""Synthesizes the cpp wrapper code and builds dynamic Python extension."""
import os
import platform
import re
import setuptools
import subprocess

import torch as th


def main():
    dirname = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(dirname, "{{name}}", "version.py")) as fid:
        try:
            __version__, = re.findall( '__version__ = "(.*)"', fid.read() )
        except:
            raise ValueError("could not find version number")

    # Build the Python extension module
    packages = setuptools.find_packages(exclude=["tests"])
    setuptools.setup(
        name="{{name}}",
        verbose=True,
        url="",
        author_email="",
        author="",
        install_requires=[
            "torch-tools",
        ],
        version=__version__,
        packages=packages
    )

if __name__ == "__main__":
    main()
