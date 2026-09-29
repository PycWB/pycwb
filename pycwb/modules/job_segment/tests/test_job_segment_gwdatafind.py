"""Tests that gwdatafind is only used when a non-empty gwdatafind block is configured."""
from types import SimpleNamespace

import pytest

import pycwb.modules.job_segment.job_segment as job_segment


def _config(gwdatafind):
    return SimpleNamespace(
        gps_start=1000000000, gps_end=1000000600, gps_center=None, superevent=None,
        dq_files=[], ifo=["L1", "H1"], segLen=600, segMLS=300, segEdge=10, segOverlap=0,
        rateANA=1024, l_high=10, inRate=16384,
        slagSize=0, slagOff=0, slagMin=0, slagMax=0,
        injection={}, frFiles=[], gwdatafind=gwdatafind, channelNamesRaw=[],
    )


@pytest.mark.parametrize("gwdatafind, expect_fetch", [
    ({}, False),
    (None, False),
    ({"frametype": ["L1_HOFT_C00", "H1_HOFT_C00"]}, True),
])
def test_gwdatafind_fetch_only_for_non_empty_block(monkeypatch, gwdatafind, expect_fetch):
    calls = []
    monkeypatch.setattr(job_segment, "job_segment_from_dq", lambda *args, **kwargs: [])
    monkeypatch.setattr(job_segment, "gwdatafind_frames_for_job_segments",
                        lambda *args, **kwargs: calls.append(args) or {})
    monkeypatch.setattr(job_segment, "attach_frame_files_to_job_segments", lambda *args, **kwargs: None)

    job_segment.create_job_segment_from_config(_config(gwdatafind))

    assert bool(calls) == expect_fetch
