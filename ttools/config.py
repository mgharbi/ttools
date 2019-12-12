"""Utilities to load configuration files."""

import os
import yaml


__all__ = ["parse_config"]

# class Config(dict):
#     def __init__(self, **kwargs):
#         for k in kwargs:
#             self[k] = kwargs[k]
#
#     def __str__(self):
#         s = ""
#         for k in self.keys():
#             s += k + ": " + str(self[k]) + "\n"
#             print(k, self[k])
#         return s

def _merge(default, user):
    """Merges two hierachical dictionaries recursively."""
    if isinstance(user, dict):
        if not isinstance(default, dict):
            raise RuntimeError("Got a dict %s to override a non-dict value %s"
                               % (user, default))
        for k in user:
            v = user[k]
            if k not in default:
                raise RuntimeError("Trying to override a parameter not provided"
                                 " in the default config: %s" % k)
            default[k] = _merge(default[k], v)
        return default
    else:
        return user


def parse_config(path, default=None):
    """Parse a .yml configuration file.
    
    See config/default.yaml for an example config with the
    possible arguments.

    Args:
        path(str): path to the config file. If none is provided, loads the
        default configuration (or returns an empty config dict)
        default(str): path to the default config file to load if path is None
    """

    if default is not None:
        with open(default) as fid:
            default = yaml.load(fid, Loader=yaml.FullLoader)
    else:
        default = {}

    if path is None:
        conf = default
    else:
        with open(path) as fid:
            conf = yaml.load(fid, Loader=yaml.FullLoader)
        conf = _merge(default, conf)

    # for section in conf:
    #     if section not in SECTIONS:
    #         raise RuntimeError("Config section '%s' not"
    #                            " recognized, should be one of"
    #                            " %s" % (section, SECTIONS))
    return conf
