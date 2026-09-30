"""Process a bounded number of local online segments without dispatching alerts.

This exercises frame reading and the worker pipeline. Use debug_run.sh to
exercise continuous acquisition and orchestration at real-time speed.
"""

import logging
from _test_integration import parser, read_segment
from pycwb.config import Config
from pycwb.modules.online.data_source import SharedMemoryDataSource
from pycwb.workflow.subflow.process_online_segment import process_online_segment


def main():
    arguments = parser()
    arguments.description = __doc__
    arguments.add_argument('--segments', type=int, default=2)
    args = arguments.parse_args()
    if args.segments < 1:
        arguments.error('--segments must be positive')
    logging.basicConfig(level=logging.INFO)
    config = Config()
    config.load_from_yaml(args.config)
    source = SharedMemoryDataSource(base_path=args.shm_base, timeout=10, poll_interval=0.1)
    source.connect()
    try:
        for index in range(args.segments):
            segment = read_segment(config, source, args.gps_start, index)
            triggers = process_online_segment(config, segment)
            print(f'Segment {index}: {len(triggers)} triggers', flush=True)
    finally:
        source.close()
    print(f'Worker pipeline passed: {args.segments} segments')


if __name__ == '__main__':
    main()
