"""
File system utils.
"""

import os
from omegaconf import OmegaConf


def f_expand(fpath):
    return os.path.expandvars(os.path.expanduser(fpath))


def f_join(*fpaths):
    """
    join file paths and expand special symbols like `~` for home dir
    """
    return f_expand(os.path.join(*fpaths))


def f_mkdir(*fpaths):
    """
    Recursively creates all the subdirs
    If exist, do nothing.
    """
    fpath = f_join(*fpaths)
    os.makedirs(fpath, exist_ok=True)
    return fpath


def omegaconf_save(cfg, *paths: str, resolve: bool = True):
    """
    Save omegaconf to yaml
    """
    OmegaConf.save(cfg, f_join(*paths), resolve=resolve)
