import orjson
import gzip
from dacite import from_dict


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
    if compress_json or output_file.endswith('.gz'):
        if not output_file.endswith('.gz'):
            output_file += '.gz'
        with gzip.open(output_file, 'wb') as f:
            f.write(orjson.dumps(dataclass_object, option=orjson.OPT_SERIALIZE_NUMPY))
    else:
        with open(output_file, 'wb') as f:
            f.write(orjson.dumps(dataclass_object, option=orjson.OPT_SERIALIZE_NUMPY))


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