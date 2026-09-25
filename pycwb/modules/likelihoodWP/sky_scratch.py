"""Compatibility exports for caller-owned sky-statistic buffers.

The arithmetic lives beside its allocating wrappers in dpf and sky_stat.
"""

from .dpf import dpf_np_loops_vec_into as dpf_np_loops_vec_into
from .sky_stat import (
    avx_GW_ps_into as avx_GW_ps_into,
    avx_ort_ps_into as avx_ort_ps_into,
    avx_stat_ps_into as avx_stat_ps_into,
)
