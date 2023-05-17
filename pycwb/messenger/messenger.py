import asyncio
import importlib
from watchfiles import awatch
import os
import re
import logging


def import_helper(name):
    p, m = name.rsplit('.', 1)

    mod = importlib.import_module(p)
    met = getattr(mod, m)
    return met

