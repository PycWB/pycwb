"""The local online example must retain candidates, including at shutdown."""

from queue import Queue
from threading import Event as StopEvent
from types import SimpleNamespace

from pycwb.modules.catalog import Catalog
from pycwb.modules.online.deduplication import TriggerDeduplicator
from pycwb.modules.online.trigger_handler import TriggerHandler
from pycwb.types.network_event import Event
from pycwb.types.online import OnlineTrigger


def candidate(gps, rho=8., segment=0):
    event = Event(id=f'event-{segment}-{gps}', job_id=segment, ndim=2,
                  ifo_list=['H1', 'L1'], time=[gps, gps], rho=[rho, rho],
                  start=[gps - .1, gps - .1], stop=[gps + .1, gps + .1],
                  low=[32., 32.], high=[480., 480.],
                  theta=[45., 45., 45., 45.], phi=[30., 30., 30., 30.])
    return OnlineTrigger(event=event, cluster={'fixture_gps': gps}, sky_stats={},
                         segment_index=segment, segment_gps=940. + 8 * segment,
                         wall_time_done=10**12)


def handler(tmp_path, queue=None, stop=None):
    config = SimpleNamespace(ifo=['H1', 'L1'], online_alert={'local_catalog': True})
    return TriggerHandler(config, queue or Queue(), stop or StopEvent(), str(tmp_path))


def test_first_online_trigger_creates_catalog_and_preserves_each_cluster(tmp_path):
    consumer = handler(tmp_path)
    for gps in [1000., 1008.]:
        consumer._save_local(candidate(gps))
    catalog = Catalog.open(str(tmp_path / 'catalog/catalog.parquet'))
    assert catalog.triggers().num_rows == 2
    assert len(list((tmp_path / 'triggers/seg_000000').glob('*/cluster.json'))) == 2


def test_shutdown_request_does_not_discard_queued_worker_results(tmp_path):
    queue, stop = Queue(), StopEvent()
    stop.set()  # Acquisition stops before in-flight workers return their results.
    queue.put(candidate(1000.))
    queue.put(candidate(1008., segment=1))
    queue.put(None)  # The manager sends this only after collecting the workers.
    handler(tmp_path, queue, stop).run()
    assert Catalog.open(str(tmp_path / 'catalog/catalog.parquet')).triggers().num_rows == 2
    assert queue.empty()


def test_manager_shutdown_flushes_buffered_triggers_when_a_worker_hangs(tmp_path, monkeypatch):
    from concurrent.futures import Future
    from pycwb.workflow import online

    monkeypatch.setattr(online, 'SHUTDOWN_WORKER_WAIT', 0.05)
    queue, stop = Queue(), StopEvent()
    consumer = handler(tmp_path, queue, stop)
    consumer.start()
    queue.put(candidate(1000.))  # Held by deduplication until the final flush.
    idle = SimpleNamespace(stop=lambda: None, close=lambda: None,
                           shutdown=lambda wait: None)
    manager = object.__new__(online.OnlineSearchManager)
    manager.__dict__.update(stop_event=stop, data_acq=idle, trigger_queue=queue,
                            trigger_handler=consumer, latency_monitor=idle,
                            executor=idle, data_source=idle)
    manager._shutdown({Future(): 'never finishes'})
    assert not consumer.is_alive()
    assert Catalog.open(str(tmp_path / 'catalog/catalog.parquet')).triggers().num_rows == 1


def test_overlapping_segments_match_native_event_arrival_times():
    dedup = TriggerDeduplicator(gps_window=.5, sky_tolerance=5.)
    weaker = candidate(1000., rho=7., segment=0)
    louder = candidate(1000.1, rho=9., segment=1)
    assert dedup.ingest(weaker) == []
    assert dedup.ingest(louder) == []
    retained = dedup.flush_all()
    assert len(retained) == 1 and retained[0] is louder
