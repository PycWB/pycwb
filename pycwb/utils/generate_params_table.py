def parse_default(param):
    if "default" in param:
        return "``{}``".format(param["default"])
    else:
        return ""

def parse_type_or_enum(param):
    if "enum" in param:
        return "``enum``"
    elif "type" in param:
        return "``{}``".format(param["type"])
    else:
        return "``unknown``"


def parse_description(param):
    if "description" in param:
        return param["description"].replace("\n", " ").replace("|", "\|")
    else:
        return "No description provided"


def generate_rst_table(params):
    # Header
    headers = [
        "   * - Parameter",
        "     - Type",
        "     - Description",
        "     - Default",
    ]

    # Format rows
    rows = [
        "\n".join(["   * - {}".format(param),
                   "     - {}".format(parse_type_or_enum(details)),
                   "     - {}".format(parse_description(details)),
                   "     - {}".format(parse_default(details)),
                   ])
        for param, details in params.items()
    ]

    # Full table
    table = "\n".join([
        ".. list-table::",
        "   :widths: 20 20 40 20",
        "   :header-rows: 1",
        "   :class: longtable",
        "",
        "\n".join(headers),
        "\n".join(rows)
    ])

    return table
