import dataclasses
import gzip

import numpy as np
import orjson
from dacite import from_dict


def _make_arrays_contiguous(obj):
    """Recursively convert all numpy arrays in *obj* to C-contiguous.

    Returns a copy of the object (dict / list / tuple) with every
    ``np.ndarray`` replaced by ``np.ascontiguousarray``.  Dataclass
    instances are first converted to dicts via :func:`dataclasses.asdict`.
    Scalars and non-array leaves are returned unchanged.
    """
    if isinstance(obj, np.ndarray):
        if obj.flags["C_CONTIGUOUS"]:
            return obj
        return np.ascontiguousarray(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _make_arrays_contiguous(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {k: _make_arrays_contiguous(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_make_arrays_contiguous(v) for v in obj)
    return obj


def save_dataclass_to_json(dataclass_object, output_file, compress_json=False):
    """
    Save dataclass object to json file

    :param dataclass_object: dataclass object
    :type dataclass_object: dataclass
    :param output_file: output file
    :type output_file: str
    :param compress_json: gzip output file, defaults to False
    :type compress_json: bool, optional
    """
    serializable = _make_arrays_contiguous(dataclass_object)
    if compress_json or output_file.endswith('.gz'):
        if not output_file.endswith('.gz'):
            output_file += '.gz'
        with gzip.open(output_file, 'wb') as f:
            f.write(orjson.dumps(serializable, option=orjson.OPT_SERIALIZE_NUMPY))
    else:
        with open(output_file, 'wb') as f:
            f.write(orjson.dumps(serializable, option=orjson.OPT_SERIALIZE_NUMPY))


def load_dataclass_from_json(dataclass_object, input_file):
    """
    Load dataclass object from json file

    :param dataclass_object: dataclass object
    :type dataclass_object: dataclass
    :param input_file: path to input json file
    :type input_file: str
    :param gzip: gzip input file, defaults to False
    :type gzip: bool, optional
    :return: dataclass object
    :rtype: dataclass
    """
    if input_file.endswith('.gz'):
        with gzip.open(input_file, 'rb') as f:
            data = orjson.loads(f.read())
    else:
        with open(input_file, 'rb') as f:
            data = orjson.loads(f.read())

    return from_dict(dataclass_object, data)