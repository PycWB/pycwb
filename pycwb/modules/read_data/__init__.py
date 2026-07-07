"""
pycwb.modules.read_data — Gravitational-wave strain data reader.

Reads GW strain from frame files (GWF), online NDS2 servers via gwpy,
or generates synthetic noise for simulations. Also supports MDC
(mock data challenge) injection I/O and data quality flag checking.
"""

from .read_data import *
from .mdc import *
from .data_check import *