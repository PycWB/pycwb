"""Read one padded online segment from explicitly supplied local frame files.

See README.md for a reproducible fake-data command. Missing inputs or imports
raise errors; this script never reports success after a failed stage.
"""

import argparse
from pathlib import Path
import time

from pycwb.config import Config
from pycwb.modules.online.data_source import SharedMemoryDataSource
from pycwb.types.online import OnlineSegment
from pycwb.workflow.subflow.process_online_segment import process_online_segment


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--config', default=str(Path(__file__).with_name('user_parameters_debug.yaml')))
    result.add_argument('--shm-base', required=True, help='Directory containing the per-detector frame folders')
    result.add_argument('--gps-start', required=True, type=float, help='GPS start of the first frame, including edge padding')
    return result


def read_segment(config, source, first_frame_gps, index=0):
    duration = float(config.online_segment_duration)
    stride = float(config.online_segment_stride)
    edge = float(config.segEdge)
    start = first_frame_gps + edge + index * stride
    # The worker removes both edges. Supply them in the payload as well.
    data = source.read_chunk(config.online_channels, start_gps=start - edge,
                             duration=duration + 2 * edge)
    assert set(data) == set(config.online_channels)
    for strain in data.values():
        assert len(strain) == round((duration + 2 * edge) * config.inRate)
    return OnlineSegment(index=index, ifos=config.ifo,
                         segment_gps_start=start, segment_gps_end=start + duration,
                         seg_edge=edge, sample_rate=config.inRate, data_payload=data,
                         wall_time_received=time.time(), stride=stride, overlap_frac=0.0)


def main():
    args = parser().parse_args()
    config = Config()
    config.load_from_yaml(args.config)
    source = SharedMemoryDataSource(base_path=args.shm_base, timeout=10, poll_interval=0.1)
    source.connect()
    try:
        assert source.is_alive()
        segment = read_segment(config, source, args.gps_start)
        assert callable(process_online_segment)
        print(f'Configuration, imports and padded frame read passed: {segment.segment_gps_start}–{segment.segment_gps_end}')
    finally:
        source.close()


if __name__ == '__main__':
    main()
