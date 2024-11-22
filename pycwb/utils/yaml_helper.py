import yaml
from jsonschema import validate, Draft202012Validator


def load_yaml(file_name, schema):
    """
    Load yaml file to ROOT global variable

    :param file_name: yaml file name
    :type file_name: str
    :param schema: user parameters schema
    :type schema: dict
    :return: full user parameters
    :rtype: dict
    """
    with open(file_name, 'r') as file:
        params = yaml.safe_load(file)

    # TODO: better error message
    # v = Draft202012Validator(schema)
    # errors = sorted(v.iter_errors(params), key=lambda e: e.path)
    # for error in errors:
    #     print(error.instance)
    #     print(error.schema_path)
    #     print(error.message)
    validate(instance=params, schema=schema)

    params = set_default(params, schema)

    return params


def set_default(params, schema):
    """
    Set default value from schema if not in params read from yaml file

    :param params: user parameters read from yaml file
    :type params: dict
    :param schema: user parameters schema
    :type schema: dict
    :return: user parameters with default value
    :rtype: dict
    """
    for key in schema['properties'].keys():
        if key not in params:
            params[key] = schema['properties'][key]['default'] if 'default' in schema['properties'][key] else None

    return params