import json


def parse_default(param):
    if "default" in param:
        try:
            default = json.dumps(param["default"])
        except TypeError:
            default = repr(param["default"])
        return "``{}``".format(default)
    else:
        return ""

def parse_type_or_enum(param):
    if "enum" in param:
        return "``enum``"
    elif "type" in param:
        return "``{}``".format(param["type"])
    else:
        return "``unknown``"


def escape_cell_text(text):
    return text.replace("|", "\\|").replace("*", "\\*")


def parse_description(param):
    if "description" in param:
        lines = [
            escape_cell_text(line.rstrip())
            for line in param["description"].splitlines()
        ]
        if len(lines) <= 1:
            return lines[0] if lines else ""
        return "\n".join(
            "| {}".format(line) if line else "|"
            for line in lines
        )
    else:
        return "No description provided"


def format_cell(prefix, content):
    lines = str(content).splitlines() or [""]
    return "\n".join(
        [prefix + lines[0]] +
        ["       " + line for line in lines[1:]]
    )


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
                   format_cell("     - ", parse_description(details)),
                   "     - {}".format(parse_default(details)),
                   ])
        for param, details in params.items()
    ]

    # Full table
    table = "\n".join([
        ".. list-table::",
        "   :widths: 18 12 52 18",
        "   :header-rows: 1",
        "   :class: longtable",
        "",
        "\n".join(headers),
        "\n".join(rows)
    ])

    return table
