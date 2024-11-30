.. _schema:

User Parameters
----------------

The default schema for user parameters is:

.. exec::
    import json
    from pycwb.constants import user_parameters_schema
    from pycwb.utils.generate_params_table import generate_rst_table, parse_description, parse_type_or_enum

    print(generate_rst_table(user_parameters_schema["properties"]))
    # json_obj = json.dumps(user_parameters_schema, sort_keys=True, indent=4)
    # print('.. code-block:: JavaScript\n\n    %s\n\n' % json_obj)