"""Jinja rendering glue for the postproduction HTML report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape


def _render_report_html(data: dict[str, Any]) -> str:
    template_dir = Path(__file__).with_name("templates")
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("postproduction_report.html.j2")
    return template.render(
        report=data,
        embedded_data_json=json.dumps(data, ensure_ascii=True),
    )
