import copy
import os
import yaml
from jsonschema import validate, Draft202012Validator


def _load_external_schema_file(file_path, base_dir):
    """
    Load a YAML schema file referenced from a pycwb_schema block.

    The *file_path* may be absolute or relative; relative paths are resolved
    against *base_dir* (normally the directory that contains the config YAML).

    The YAML file must contain a single top-level mapping that is a valid JSON
    schema.  A ``pycwb_schema`` key inside the external file is **not**
    processed recursively.

    :param file_path: path to the external schema YAML file
    :type file_path: str
    :param base_dir: directory used to resolve relative paths
    :type base_dir: str
    :return: schema dict loaded from the file
    :rtype: dict
    """
    if not os.path.isabs(file_path):
        file_path = os.path.join(base_dir, file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"External schema file not found: {file_path}"
        )
    with open(file_path, 'r') as fh:
        schema = yaml.safe_load(fh)
    if not isinstance(schema, dict):
        raise ValueError(
            f"External schema file must contain a YAML mapping (dict), got: "
            f"{type(schema).__name__} in {file_path}"
        )
    return schema


def resolve_schema(raw_params, default_schema, base_dir=None):
    """
    Resolve the effective schema from raw YAML parameters and a default schema.

    If ``raw_params`` contains a ``pycwb_schema`` key its value controls how
    the schema is built.  The schema definition may be provided **inline** in
    the config YAML or via an **external YAML file** referenced by
    ``schema_file``.

    Supported modes
    ---------------
    * ``mode: extend`` (default) – merges additional field definitions on top
      of the default schema, preserving all default fields while adding or
      overriding individual entries.  Supply extra fields via inline
      ``properties`` *or* an external file with ``schema_file``.
    * ``mode: replace`` – uses the provided definition as the complete schema,
      completely replacing the default.  Supply the full schema via inline
      ``schema`` *or* an external file with ``schema_file``.

    External schema file
    --------------------
    The optional ``schema_file`` key points to a YAML file (absolute path or
    relative to the config YAML directory) whose contents are loaded as the
    schema definition.  For ``extend`` mode the file should be a JSON-schema
    ``properties`` mapping; for ``replace`` mode it should be a complete JSON
    schema object.

    Examples
    --------
    Inline extend::

        pycwb_schema:
          mode: extend          # optional, 'extend' is the default
          properties:
            my_custom_param:
              type: string
              description: "A custom parameter"
              default: "my_default"

    External extend (file contains only the extra ``properties`` dict)::

        pycwb_schema:
          mode: extend
          schema_file: ./my_extra_fields.yaml

    External replace (file contains a full JSON schema)::

        pycwb_schema:
          mode: replace
          schema_file: ./my_full_schema.yaml

    Inline replace::

        pycwb_schema:
          mode: replace
          schema:
            type: object
            properties: { ... }
            required: []
            additionalProperties: true

    :param raw_params: raw parameters loaded from the YAML file
    :type raw_params: dict
    :param default_schema: the default JSON schema to fall back on
    :type default_schema: dict
    :param base_dir: directory used to resolve relative ``schema_file`` paths;
        defaults to the current working directory
    :type base_dir: str or None
    :return: effective JSON schema to use for validation and default-filling
    :rtype: dict
    """
    if 'pycwb_schema' not in raw_params:
        return default_schema

    pycwb_schema_def = raw_params['pycwb_schema']
    if not isinstance(pycwb_schema_def, dict):
        raise ValueError("'pycwb_schema' must be a mapping (dict)")

    mode = pycwb_schema_def.get('mode', 'extend')
    schema_file = pycwb_schema_def.get('schema_file')
    _base = base_dir or os.getcwd()

    if mode == 'replace':
        if schema_file is not None:
            replacement = _load_external_schema_file(schema_file, _base)
        else:
            replacement = pycwb_schema_def.get('schema')
        if replacement is None:
            raise ValueError(
                "pycwb_schema with mode='replace' must provide either a "
                "'schema' key (inline) or a 'schema_file' key (external file)"
            )
        return replacement

    if mode == 'extend':
        if schema_file is not None:
            extra_properties = _load_external_schema_file(schema_file, _base)
        else:
            extra_properties = pycwb_schema_def.get('properties', {})
        if not extra_properties:
            return default_schema
        effective = copy.deepcopy(default_schema)
        effective['properties'].update(extra_properties)
        # When additionalProperties is False the new fields must be listed;
        # switching to True is the safest approach when user adds custom keys.
        if effective.get('additionalProperties') is False:
            effective['additionalProperties'] = True
        return effective

    raise ValueError(
        f"Unknown pycwb_schema mode: {mode!r}. Supported modes are 'extend' and 'replace'."
    )


def load_yaml(file_name, schema):
    """
    Load yaml file and validate/complete it against *schema*.

    If the YAML file contains a top-level ``pycwb_schema`` key the schema is
    first resolved via :func:`resolve_schema` before validation.  The
    ``pycwb_schema`` key is then stripped from the returned parameters dict.
    Relative paths inside ``pycwb_schema.schema_file`` are resolved relative
    to the directory containing *file_name*.

    :param file_name: yaml file name
    :type file_name: str
    :param schema: default user parameters schema
    :type schema: dict
    :return: full user parameters
    :rtype: dict
    """
    with open(file_name, 'r') as file:
        params = yaml.safe_load(file)

    # Resolve effective schema from any pycwb_schema metadata in the YAML file,
    # then remove the metadata key so it does not reach validation.
    base_dir = os.path.dirname(os.path.abspath(file_name))
    effective_schema = resolve_schema(params, schema, base_dir=base_dir)
    params.pop('pycwb_schema', None)

    # TODO: better error message
    # v = Draft202012Validator(effective_schema)
    # errors = sorted(v.iter_errors(params), key=lambda e: e.path)
    # for error in errors:
    #     print(error.instance)
    #     print(error.schema_path)
    #     print(error.message)
    validate(instance=params, schema=effective_schema)

    params = set_default(params, effective_schema)

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