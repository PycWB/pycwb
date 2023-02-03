import yaml
from jsonschema import validate, Draft202012Validator
from pycwb.constants import user_parameters_schema
from ROOT import gROOT


# v = Draft202012Validator(schema)
# errors = sorted(v.iter_errors(params), key=lambda e: e.path)
# for error in errors:
#     print(error.instance)
#     print(error.schema_path)
#     print(error.message)

def load_yaml(file_name, load_to_root=True):
    """
    Load yaml file to ROOT global variable
    :param gROOT:
    :param file_name:
    :return:
    """
    with open(file_name, 'r') as file:
        params = yaml.safe_load(file)

    # TODO: better error message
    validate(instance=params, schema=user_parameters_schema)

    if load_to_root:
        params = add_generated_key(params)
        # assign variable
        cmd = ""
        for key in params.keys():
            cmd += assign_variable(user_parameters_schema, key, params[key]) + '\n'

            gROOT.ProcessLine(cmd)
    else:
        params = set_default(params, user_parameters_schema)

    return params


def set_default(params, schema):
    """
    Set default value if not set
    :param params:
    :param schema:
    :return:
    """
    for key in schema['properties'].keys():
        if key not in params:
            params[key] = schema['properties'][key]['default']

    return params


def assign_variable(schema, key, value):
    """
    Assign variable
    :param schema:
    :param key:
    :param value:
    :return:
    """
    cmd = ""

    if "c_type" in schema['properties'][key]:
        return process_special_type(schema['properties'][key]['c_type'], key, value)

    if isinstance(value, str):
        if len(value) == 1:
            cmd = f"{key} = '{value}';"
        else:
            cmd = f"strcpy({key},\"{value}\");"
    elif isinstance(value, bool):
        cmd = f"{key} = {'true' if value else 'false'};"
    elif isinstance(value, int) or isinstance(value, float):
        cmd = f"{key} = {value};"
    elif isinstance(value, list):
        for i, v in enumerate(value):
            if isinstance(v, str):
                cmd += f"strcpy({key}[{i}],\"{v}\");\n"
            elif isinstance(v, bool):
                cmd += f"{key}[{i}] = {'true' if value else 'false'};\n"
            elif isinstance(v, int) or isinstance(v, float):
                cmd += f"{key}[{i}] = {v};\n"

    if not cmd:
        print("Error in:", key, value)

    return cmd


def add_generated_key(params):
    """
    Add generated key to the user parameters for ROOT
    :param params:
    :return:
    """
    new_params = {}
    for key in params.keys():
        if key == 'ifo':
            new_params['nIFO'] = len(params[key])
        if key == 'DQF':
            new_params['nDQF'] = len(params[key])

        new_params[key] = params[key]

    return new_params


def process_special_type(c_type, key, value):
    """
    Process special type
    :param c_type:
    :param key:
    :param value:
    :return:
    """
    if c_type == 'TMacro':
        return f'{key} = TMacro("{value}");'
    if c_type == 'dqfile':
        arr = ',\n'.join([f'{{ "{v[0]}", "{v[1]}", {v[2]}, {v[3]}, {v[4]}, {v[5]}}}' for v in value])
        str = f'dqfile dqf[{len(value)}]={{ \n{arr} \n}}; \nfor(int i=0;i<nDQF;i++) DQF[i]=dqf[i]; \n'
        return str
