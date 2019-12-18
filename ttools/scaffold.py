"""Routines to initialize a repo from templates."""
import os
import shutil
import subprocess

from pkg_resources import resource_filename

from jinja2 import Environment, PackageLoader, select_autoescape


def init_repo(path):
    print("### ttools project initializer ###")

    name = input("\nPlease enter a package name: ")

    if path is None:
        path = os.path.join(os.curdir, name)
    path = os.path.abspath(path)

    if os.path.exists(path):
        print("\nThe path '%s' already exists, aborting!" % path)
        return

    print("\nCreating Python package '%s' in %s\n" % (name, path))

    subfolders = sorted([name, "config", "data", "output", "scripts", "tests",
                         "checkpoints"])

    print("Setting up folder structure:")
    print("+%s" % os.path.basename(path))
    os.makedirs(path, exist_ok=True)
    for idx, s in enumerate(subfolders):
        if idx == len(subfolders)-1:
            print("  └─ %s" % s)
        else:
            print("  ├─ %s" % s)

        os.makedirs(os.path.join(path, s), exist_ok=True)

    env = Environment(
        loader=PackageLoader('ttools', 'templates'),
    )

    print("Initializing repo files")
    for f in ["setup.py", "Makefile", "README.md", ".gitignore", "pytest.ini"]:
        # resource_filename(__name__, os.path.join('templates', f))
        template = env.get_template(f)
        dst = os.path.join(path, f)
        with open(dst, 'w') as fid:
            fid.write(template.render(name=name))

    for f in ["models.py", "datasets.py", "interfaces.py", "callbacks.py",
              "version.py", "__init__.py"]:
        # resource_filename(__name__, os.path.join('templates', f))
        template = env.get_template(f)
        dst = os.path.join(path, name, f)
        with open(dst, 'w') as fid:
            fid.write(template.render(name=name))

    for f in ["train.py", "eval.py"]:
        # resource_filename(__name__, os.path.join('templates', f))
        template = env.get_template(f)
        dst = os.path.join(path, "scripts", f)
        with open(dst, 'w') as fid:
            fid.write(template.render(name=name))

    for f in ["test_basic.py"]:
        # resource_filename(__name__, os.path.join('templates', f))
        template = env.get_template(f)
        dst = os.path.join(path, "tests", f)
        with open(dst, 'w') as fid:
            fid.write(template.render(name=name))

    fname = resource_filename(__name__, os.path.join('templates', "default.yml"))
    dst = os.path.join(path, "config", "default.yml")
    shutil.copy(fname, dst)

    print("Done! Check the readme at %s" % os.path.join(path, "README.md"))
