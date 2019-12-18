"""Executable entrypoints."""
import os
import argparse
import imageio

import ttools.scaffold as scaffold

import skimage.transform as xform


def resize():
    parser = argparse.ArgumentParser(description="upsample an image")
    parser.add_argument("image")
    parser.add_argument("factor", type=float, help="upsampling factor")
    parser.add_argument("output")
    args = parser.parse_args()

    im = imageio.imread(args.image)
    im = xform.rescale(im, args.factor, order=0)
    imageio.imsave(args.output, im)


def im2vid():
    parser = argparse.ArgumentParser(
        description="converts a series of image frames to a video.")
    parser.add_argument("images", nargs="+")
    parser.add_argument("output")
    args = parser.parse_args()

    images = []
    print("loading", len(args.images), "images")
    for filename in sorted(args.images):
        images.append(imageio.imread(filename))
        print(".")
    print("saving video")
    imageio.mimsave(args.output, images)


def new_project():
    parser = argparse.ArgumentParser(
        description="initialized a directory for a new project.")
    parser.add_argument("--path", help="path to the root of the repo.")
    args = parser.parse_args()

    scaffold.init_repo(args.path)
