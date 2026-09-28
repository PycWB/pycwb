"""Load configuration-local detector definitions and reproducible snapshots."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from jsonschema import validate

from pycwb.constants.detectors import DETECTOR_GEOMETRIES


def _object(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


_NUMBER = {"type": "number"}
_ANGLE = {"type": "number", "minimum": -math.pi / 2, "maximum": math.pi / 2}
_ARM = _object({"az": _NUMBER, "alt": _ANGLE,
                "midpoint": {"type": "number", "exclusiveMinimum": 0}})
_PARAMETERS = _object({
    "name": {"type": "string", "minLength": 1},
    "lat": _ANGLE,
    "lon": {"type": "number", "minimum": -math.pi, "maximum": math.pi},
    "elevation": _NUMBER, "x": _ARM, "y": _ARM,
})
_ENTRY = _object({"detector": {"type": "string", "pattern": r"^[A-Za-z][A-Za-z0-9_]*$"},
                  "parameters": _PARAMETERS})
_SCHEMA = _object({
    "schema_version": {"type": "integer", "const": 1},
    "geometries": {"type": "object", "additionalProperties": _ENTRY},
})


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate detector-definition JSON key: {key}")
        result[key] = value
    return result


def _finite(value):
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Detector definitions require finite numeric values")


def _validate_entry(geometry_id, entry, *, bundled=False):
    schema = deepcopy(_ENTRY)
    if bundled:
        for arm in ("x", "y"):
            schema["properties"]["parameters"]["properties"][arm]["properties"]["midpoint"] = {"type": "number", "minimum": 0}
    validate(entry, schema)
    _finite(entry)
    prefix, separator, suffix = geometry_id.partition(":")
    if prefix != entry["detector"] or not separator or not suffix or ":" in suffix:
        raise ValueError(f"Geometry ID {geometry_id!r} must be '<detector>:<definition>'")
    # Collinear arms do not define a two-arm interferometer. Non-orthogonal
    # detectors (e.g. triangular observatories) are deliberately supported.
    arms = entry["parameters"]
    x, y = arms["x"], arms["y"]
    dot = (math.cos(x["alt"]) * math.cos(y["alt"]) * math.cos(x["az"] - y["az"])
           + math.sin(x["alt"]) * math.sin(y["alt"]))
    if not bundled and abs(dot) >= 1 - 1e-12:
        raise ValueError(f"Geometry {geometry_id!r} has collinear arms")


def load_detector_definitions(file_name, yaml_file):
    """Return an independent registry and JSON-file provenance.

    Geographic definitions replace whole entries; aliases are selectors, not
    definition IDs. Input files cannot inject fixed-vector or source semantics.
    """
    registry = deepcopy(DETECTOR_GEOMETRIES)
    if not file_name:
        return registry, {}
    path = Path(file_name)
    if not path.is_absolute():
        path = Path(yaml_file).resolve().parent / path
    raw = path.read_bytes()
    document = json.loads(raw, object_pairs_hook=_unique_object)
    validate(document, _SCHEMA)
    _finite(document)
    from pycwb.constants.detectors import DETECTOR_GEOMETRY_ALIASES
    for geometry_id, entry in document["geometries"].items():
        _validate_entry(geometry_id, entry)
        if geometry_id in DETECTOR_GEOMETRY_ALIASES:
            raise ValueError(f"Use canonical ID {DETECTOR_GEOMETRY_ALIASES[geometry_id]!r}, not alias {geometry_id!r}")
        registry[geometry_id] = {**deepcopy(entry), "source": "custom", "vectors": None}
    return registry, {
        "path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
        "schema_version": 1,
        "overridden_ids": sorted(set(document["geometries"]) & set(DETECTOR_GEOMETRIES)),
    }


def restore_detector_registry(snapshot):
    """Validate and copy saved definitions without consulting an external file."""
    if not isinstance(snapshot, dict):
        raise ValueError("detector_registry snapshot must be a mapping")
    for geometry_id, entry in snapshot.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid geometry snapshot for {geometry_id}")
        _validate_entry(geometry_id, {key: entry.get(key) for key in ("detector", "parameters")},
                        bundled=entry.get("source") in ("lal", "cwb"))
        if entry.get("source") not in ("lal", "cwb", "custom"):
            raise ValueError(f"Invalid geometry source for {geometry_id}")
        vectors = entry.get("vectors")
        if vectors is not None:
            if entry["source"] != "cwb" or len(vectors) != 3 or any(len(v) != 3 for v in vectors):
                raise ValueError(f"Invalid fixed vectors for {geometry_id}")
            _finite(vectors)
        elif entry["source"] == "cwb":
            raise ValueError(f"Missing fixed vectors for {geometry_id}")
    return deepcopy(snapshot)
