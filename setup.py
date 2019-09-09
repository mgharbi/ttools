"""Build and install script for ttools."""
import re
import setuptools


with open('ttools/version.py') as fid:
    try:
        __version__, = re.findall( '__version__ = "(.*)"', fid.read() )
    except:
        raise ValueError("could not find version number")


with open("README.rst", "r") as fh:
    long_description = fh.read()

packages = setuptools.find_packages(exclude=["tests"])

docs_require = ["sphinx"]
tests_require = ["pytest"]

setuptools.setup(
    name="torch-tools",
    version=__version__,
    scripts=[],
    author="Michaël Gharbi",
    author_email="mgharbi@adobe.com",
    description="A library of helpers to train, evaluate and visualize deep nets with PyTorch.",
    long_description=long_description,
    url="https://github.com/mgharbi/ttools",
    packages=packages,
    include_package_data=True,
    license="MIT",
    install_requires=[
        "torch",
        "tqdm",
        "numpy",
        "torchvision",
        "coloredlogs",
        "visdom",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
    ],
    extras_require={
        "docs": docs_require,
        "tests": tests_require,
        "dev": docs_require + tests_require,
    }
)
