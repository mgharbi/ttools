"""Build and install script for ttools."""
import re
import setuptools


with open('ttools/version.py') as fid:
    try:
        __version__, = re.findall( '__version__ = "(.*)"', fid.read() )
    except:
        raise ValueError("could not find version number")


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="ttools",
    version=__version__,
    scripts=[],
    author="Michaël Gharbi",
    author_email="mgharbi@adobe.com",
    description="A library of helpers to train, evaluate and visualize deep nets with PyTorch.",
    long_description=long_description,
    url="https://github.com/mgharbi/ttools",
    packages=["ttools"],
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
    ],
)
