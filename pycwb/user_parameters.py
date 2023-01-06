import yaml
from jsonschema import validate, Draft202012Validator
from pycWB.config.user_parameters_schema import schema
from pycWB.config.constants import CWB_CAT


# v = Draft202012Validator(schema)
# errors = sorted(v.iter_errors(params), key=lambda e: e.path)
# for error in errors:
#     print(error.instance)
#     print(error.schema_path)
#     print(error.message)

def load_yaml(gROOT, file_name):
    with open(file_name, 'r') as file:
        params = yaml.safe_load(file)

    # TODO: better error message
    validate(instance=params, schema=schema)

    params = add_generated_key(params)
    # TODO: assign to ROOT
    cmd = ""
    for key in params.keys():
        cmd += assign_variable(schema, key, params[key]) + '\n'

    gROOT.ProcessLine(cmd)

    return params


def assign_variable(schema, key, value):
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
    new_params = {}
    for key in params.keys():
        if key == 'ifo':
            new_params['nIFO'] = len(params[key])
        if key == 'DQF':
            new_params['nDQF'] = len(params[key])

        new_params[key] = params[key]
    return new_params


def process_special_type(c_type, key, value):
    if c_type == 'TMacro':
        return f'{key} = TMacro("{value}");'
    if c_type == 'dqfile':
        arr = ',\n'.join([f'{{ "{v[0]}", "{v[1]}", {v[2]}, {v[3]}, {v[4]}, {v[5]}}}' for v in value])
        str = f'dqfile dqf[{len(value)}]={{ \n{arr} \n}}; \nfor(int i=0;i<nDQF;i++) DQF[i]=dqf[i]; \n'
        return str
