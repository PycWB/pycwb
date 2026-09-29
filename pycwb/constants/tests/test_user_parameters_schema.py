"""Tests for the structure of pycwb.constants.user_parameters_schema."""
from pycwb.constants.user_parameters_schema import schema

# Keywords used by the property definitions in user_parameters_schema.py:
# JSON-schema keywords plus the pycWB extensions ``cwb``, ``category`` and ``c_type``.
# Extend this set when a new keyword is intentionally introduced.
ALLOWED_PROPERTY_KEYWORDS = {
    "type",
    "description",
    "default",
    "enum",
    "minimum",
    "maximum",
    "items",
    "uniqueItems",
    "additionalProperties",
    "properties",
    "allOf",
    "minLength",
    "cwb",
    "category",
    "c_type",
}


def _unknown_keywords(properties, path=()):
    """Yield (property path, keyword) for every keyword not in ALLOWED_PROPERTY_KEYWORDS."""
    for name, definition in properties.items():
        prop_path = path + (name,)
        for keyword in definition:
            if keyword not in ALLOWED_PROPERTY_KEYWORDS:
                yield ".".join(prop_path), keyword
        items = definition.get("items")
        if isinstance(items, dict):
            yield from _unknown_keywords({"items": items}, prop_path)


def test_schema_properties_use_only_known_keywords():
    # Catches e.g. a stray comma inside a multi-line description, which turns the
    # following string literals (implicitly concatenated) into a bogus dict key.
    unknown = list(_unknown_keywords(schema["properties"]))
    assert unknown == [], f"unknown keywords in schema properties: {unknown}"


def test_gwdatafind_has_empty_default_and_full_description():
    gwdatafind = schema["properties"]["gwdatafind"]
    assert gwdatafind["default"] == {}
    for key in ("site", "frametype", "host", "urltype"):
        assert key in gwdatafind["description"]
    assert "datafind.igwn.org" in gwdatafind["description"]
