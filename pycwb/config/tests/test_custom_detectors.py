"""External geometry definitions, isolation, physical response and restoration."""
import hashlib
import json

import numpy as np
import orjson
import pytest
from jsonschema import ValidationError

from pycwb.config import Config
from pycwb.config.detector_definitions import load_detector_definitions
from pycwb.constants.detectors import DETECTOR_GEOMETRIES
from pycwb.modules.catalog.catalog import _build_schema_metadata
from pycwb.types.detector import Detector, DetectorNetwork, compute_sky_delay_and_patterns
from pycwb.utils.network import max_delay


@pytest.fixture
def document():
    return {"schema_version": 1, "geometries": {
        "X1:custom-v1": {"detector": "X1", "parameters": {
            "name": "Example", "lat": 0.0, "lon": 0.0, "elevation": 0.0,
            "x": {"az": 0.0, "alt": 0.0, "midpoint": 2000.0},
            "y": {"az": np.pi / 2, "alt": 0.0, "midpoint": 2000.0},
        }}
    }}


@pytest.fixture
def config_file(tmp_path, monkeypatch, document):
    # Bypass unrelated filter-file checks, retaining real derived geometry.
    for method in ("check_xtalk_file", "check_MRA_catalog",
                   "check_lagStep", "check_analyze_injection_only"):
        monkeypatch.setattr(Config, method, lambda *args: None)
    monkeypatch.setattr(Config, "MRAcatalog", "", raising=False)
    (tmp_path / "detectors.json").write_text(json.dumps(document))
    path = tmp_path / "user_parameters.yaml"
    path.write_text("analysis: 2G\nifo: [H1, X1]\nrefIFO: H1\n"
                    "detector_definitions_file: detectors.json\n"
                    "detector_geometry:\n  X1: X1:custom-v1\n")
    return path


def test_custom_geometry_physics_and_catalog_snapshot(config_file, monkeypatch):
    # Relative paths must not depend on the process working directory.
    monkeypatch.chdir(config_file.parent.parent)
    cfg = Config()
    cfg.load_from_yaml(config_file)
    det = cfg.get_detector("X1")
    direct = Detector("X1", "Example", 0, 0, 0, 0, 0, 2000, np.pi / 2, 0, 2000)
    np.testing.assert_allclose(det.response, direct.response, atol=1e-15)
    # Independent analytic limit: at lat=lon=0, north is +Z, east is +Y.
    np.testing.assert_allclose(det.vertex_vec_earth_centered, [6378137, 0, 0], atol=1e-8)
    np.testing.assert_allclose(det.response, np.diag([0, -0.5, 0.5]), atol=1e-15)
    h1 = Detector("H1")
    expected_delay = np.linalg.norm(det.vertex_vec_earth_centered - h1.vertex_vec_earth_centered) / 299792458
    assert cfg.max_delay == pytest.approx(expected_delay, abs=1e-15)
    assert max_delay(cfg.detectors) == pytest.approx(expected_delay, abs=1e-15)
    args = (cfg.detectors, "H1", 4096, 32, 1126259462)
    actual = compute_sky_delay_and_patterns(*args, n_sky=12)
    reference = compute_sky_delay_and_patterns([h1, direct], *args[1:], n_sky=12)
    for a, b in zip(actual, reference):
        np.testing.assert_allclose(a, b, atol=1e-14)
    assert DetectorNetwork(detectors=[det])._get_detector_info()[0]["code"] == "X1"
    source = config_file.parent / "detectors.json"
    assert cfg.detector_definitions_provenance["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = orjson.loads(_build_schema_metadata(cfg)[b"config"])
    source.unlink()
    restored = Config()
    restored.load_from_dict(metadata)
    restored_det = restored.get_detector("X1")
    np.testing.assert_array_equal(restored_det.response, det.response)
    restored.detector_registry["X1:custom-v1"]["parameters"]["lat"] = 1
    assert cfg.detector_registry["X1:custom-v1"]["parameters"]["lat"] == 0
    assert "X1:custom-v1" not in DETECTOR_GEOMETRIES


def test_override_is_complete_local_and_provenanced(config_file, document):
    entry = document["geometries"].pop("X1:custom-v1")
    entry["detector"] = "H1"
    document["geometries"]["H1:lal@pycwb-1"] = entry
    source = config_file.parent / "detectors.json"
    source.write_text(json.dumps(document))
    registry, provenance = load_detector_definitions(source, config_file)
    custom = Detector("H1", geometry_registry=registry)
    assert custom.latitude == 0
    assert Detector("H1").latitude != 0
    assert provenance["overridden_ids"] == ["H1:lal@pycwb-1"]
    # Replacing a cWB ID with geographic parameters must not retain its vectors
    # or activate cWB-specific event export precision.
    document["geometries"] = {"H1:cwb": entry}
    source.write_text(json.dumps(document))
    registry, _ = load_detector_definitions(source, config_file)
    custom = Detector("H1:cwb", geometry_registry=registry)
    assert custom.geometry_model == "custom"
    assert custom.latitude == 0


@pytest.mark.parametrize("change", [
    lambda d: d.update(schema_version=2),
    lambda d: d["geometries"]["X1:custom-v1"].update(detector="Y1"),
    lambda d: d["geometries"]["X1:custom-v1"]["parameters"].update(lat=2),
    lambda d: d["geometries"]["X1:custom-v1"]["parameters"].update(lon=float("nan")),
    lambda d: d["geometries"]["X1:custom-v1"]["parameters"]["x"].update(midpoint=0),
    lambda d: d["geometries"]["X1:custom-v1"]["parameters"]["y"].update(az=0),
    lambda d: d["geometries"]["X1:custom-v1"]["parameters"].pop("elevation"),
    lambda d: d["geometries"]["X1:custom-v1"].update(vectors=[]),
])
def test_invalid_definitions(tmp_path, document, change):
    change(document)
    path = tmp_path / "detectors.json"
    path.write_text(json.dumps(document))
    with pytest.raises((ValueError, ValidationError)):
        load_detector_definitions(path, tmp_path / "params.yaml")


def test_duplicate_keys_and_missing_file(tmp_path):
    path = tmp_path / "detectors.json"
    path.write_text('{"schema_version":1,"schema_version":1,"geometries":{}}')
    with pytest.raises(ValueError, match="Duplicate"):
        load_detector_definitions(path, path)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        load_detector_definitions(path, path)


def test_new_detector_requires_selection(config_file):
    config_file.write_text(config_file.read_text().replace("  X1: X1:custom-v1", "  H1: H1:lal"))
    with pytest.raises(ValueError, match="Unknown detector geometry"):
        Config().load_from_yaml(config_file)


def test_restore_requires_snapshot_for_external_definitions():
    with pytest.raises(ValueError, match="snapshot"):
        Config().load_from_dict({"ifo": ["H1"], "detector_definitions_file": "missing.json"})


def test_custom_injection_matches_equivalent_builtin(tmp_path, document):
    from copy import deepcopy
    from pycwb.modules.injection.strain import project_to_detector
    from pycwb.types.time_series import TimeSeries

    document["geometries"]["X1:custom-v1"]["parameters"] = deepcopy(
        DETECTOR_GEOMETRIES["H1:lal@pycwb-1"]["parameters"]
    )
    path = tmp_path / "detectors.json"
    path.write_text(json.dumps(document))
    registry, _ = load_detector_definitions(path, path)
    hp = TimeSeries(data=np.arange(32, dtype=float), dt=1 / 4096, t0=-1)
    hc = TimeSeries(data=np.ones(32), dt=1 / 4096, t0=-1)
    h1, custom = project_to_detector(
        hp, hc, 0.7, -0.2, 0.3,
        [Detector("H1"), Detector("X1:custom-v1", geometry_registry=registry)], 1126259462,
    )
    np.testing.assert_array_equal(custom.data, h1.data)
    assert custom.t0 == h1.t0


def test_legacy_metadata_and_config_reuse(config_file):
    cfg = Config()
    cfg.load_from_yaml(config_file)
    cfg.load_from_dict({"ifo": ["H1"], "detector_geometry": {}})
    assert "X1:custom-v1" not in cfg.detector_registry
    assert cfg.detector_geometry == {"H1": "H1:lal@pycwb-1"}


def test_alias_definition_rejected(config_file, document):
    entry = document["geometries"].pop("X1:custom-v1")
    entry["detector"] = "H1"
    document["geometries"]["H1:lal"] = entry
    source = config_file.parent / "detectors.json"
    source.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="canonical ID"):
        load_detector_definitions(source, config_file)


def test_analysis_reuses_instances_without_constructing_detectors(config_file, monkeypatch):
    from pycwb.modules.injection.strain import project_to_detector
    from pycwb.modules.likelihoodWP.pixel_data import build_sky_delay_and_antenna_patterns
    from pycwb.modules.super_cluster_native.super_cluster import setup_supercluster
    from pycwb.types.time_series import TimeSeries
    from pycwb.types.job import WaveSegment
    from pycwb.types.network_cluster import Cluster, ClusterMeta
    from pycwb.types.network_event import Event
    from pycwb.types.pixel_arrays import PixelArrays

    calls = []
    original = Detector.__init__

    def counted(self, *args, **kwargs):
        calls.append(args[0])
        original(self, *args, **kwargs)

    monkeypatch.setattr(Detector, "__init__", counted)
    cfg = Config()
    cfg.load_from_yaml(config_file)
    assert calls == cfg.ifo
    assert cfg.get_detectors() is cfg.detectors
    assert cfg.get_detector("X1") is cfg.detectors[1]
    assert cfg.get_detectors(["X1", "H1"])[0] is cfg.detectors[1]
    with pytest.raises(ValueError, match="not active"):
        cfg.get_detector("V1")

    def unexpected(*args, **kwargs):
        raise AssertionError("Analysis must reuse initialized detectors")

    monkeypatch.setattr(Detector, "__init__", unexpected)
    cfg.healpix = cfg.MIN_SKYRES_HEALPIX = 1
    gps = 1126259462
    sky = setup_supercluster(cfg, gps)
    hp = TimeSeries(data=np.ones(32), dt=1 / 4096, t0=gps)
    result = build_sky_delay_and_antenna_patterns(2, [hp, hp], cfg)
    for actual, key in zip(result, ("ml_likelihood", "FP_likelihood", "FX_likelihood")):
        np.testing.assert_array_equal(actual, sky[key])
    assert len(project_to_detector(hp, hp, 0.7, -0.2, 0.3, cfg.detectors, 0)) == 2
    assert max_delay(cfg.detectors) == cfg.max_delay

    job = WaveSegment(index=1, ifos=cfg.ifo, analyze_start=gps, analyze_end=gps + 1200,
                      sample_rate=4096, seg_edge=10, shift=None)
    pixels = PixelArrays.from_arrays(
        time=np.array([160, 176]), frequency=np.array([2, 2]), layers=np.array([16, 16]),
        rate=np.array([32., 32.]), noise_rms=np.ones((2, 2)),
        pixel_index=np.array([[160, 176], [160, 176]]), n_ifo=2,
        core=np.ones(2, dtype=bool), likelihood=np.ones(2),
    )
    meta = ClusterMeta()
    meta.reconstructed_theta, meta.reconstructed_phi = 45., 20.
    event = Event()
    event.output_py(job, Cluster(pixel_arrays=pixels, cluster_meta=meta), cfg)
    assert len(event.bp) == len(event.bx) == 2
    assert np.isfinite(event.time).all()


def test_runtime_instances_excluded_from_both_catalog_formats(config_file, tmp_path):
    import pickle
    from pycwb.modules.catalog.catalog_json import JSONCatalog

    cfg = Config()
    cfg.load_from_yaml(config_file)
    payload = cfg.to_dict()
    assert "_detectors" not in payload and "_detectors_by_name" not in payload
    payload["detector_registry"]["X1:custom-v1"]["parameters"]["lat"] = 1
    assert cfg.get_detector("X1").latitude == 0
    # The legacy JSON implementation lacks two newer progress methods. Supply
    # those unused abstract methods solely to exercise its real metadata writer.
    class MetadataCatalog(JSONCatalog):
        def add_lag_progress(self, *args, **kwargs):
            raise NotImplementedError

        def get_completed_lags(self, *args, **kwargs):
            raise NotImplementedError

    json_catalog = MetadataCatalog.create(str(tmp_path / "catalog.json"), cfg, [])
    parquet_payload = orjson.loads(_build_schema_metadata(cfg)[b"config"])
    (config_file.parent / "detectors.json").unlink()
    for metadata in (json_catalog.config, parquet_payload):
        assert "_detectors" not in metadata and "_detectors_by_name" not in metadata
        restored = Config()
        restored.load_from_dict(metadata)
        assert restored.get_detector("X1") is restored.detectors[1]
        assert restored.get_detector("X1") is not cfg.get_detector("X1")
        np.testing.assert_array_equal(restored.get_detector("X1").response, cfg.get_detector("X1").response)
    # Multiprocessing pickle transport retains shared identity inside a worker.
    worker = pickle.loads(pickle.dumps(cfg))
    assert worker.get_detector("X1") is worker.detectors[1]
    assert worker.get_detector("X1") is not cfg.get_detector("X1")


def test_reload_rebuilds_instances(config_file):
    cfg = Config()
    cfg.load_from_yaml(config_file)
    old = cfg.get_detector("X1")
    source = config_file.parent / "detectors.json"
    document = json.loads(source.read_text())
    document["geometries"]["X1:custom-v1"]["parameters"]["lat"] = 0.4
    source.write_text(json.dumps(document))
    cfg.load_from_yaml(config_file)
    assert cfg.get_detector("X1") is not old
    assert cfg.get_detector("X1").latitude == 0.4
    assert old.latitude == 0


def test_injection_generation_passes_shared_instances_in_job_order(config_file, monkeypatch):
    from pycwb.modules.injection import strain
    from pycwb.types.time_series import TimeSeries

    cfg = Config()
    cfg.load_from_yaml(config_file)
    hp = TimeSeries(data=np.ones(32), dt=1 / 4096, t0=0)
    monkeypatch.setattr(strain, "import_function", lambda name: lambda **kwargs: {
        "type": "polarizations", "hp": hp, "hc": hp,
    })
    seen = []
    project = strain.project_to_detector

    def capture(hp, hc, ra, dec, pol, detectors, gps):
        seen.extend(detectors)
        return project(hp, hc, ra, dec, pol, detectors, gps)

    monkeypatch.setattr(strain, "project_to_detector", capture)
    result = strain.generate_strain_from_injection(
        {"generator": "example.waveform", "ra": 0.7, "dec": -0.2, "pol": 0.3,
         "gps_time": 1126259462}, cfg, 4096, ["X1", "H1"],
    )
    assert len(result) == 2
    assert seen[0] is cfg.get_detector("X1")
    assert seen[1] is cfg.get_detector("H1")
