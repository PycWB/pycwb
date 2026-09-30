"""Generate references from the installed checkout before every Sphinx build."""

from pathlib import Path


def generate(app):
    """Use the same API generation and command registry on CI and Read the Docs."""
    from sphinx.ext.apidoc import main as apidoc

    from pycwb.cli.main import COMMANDS, create_parser

    source = Path(app.srcdir)
    package = source.parents[1] / "pycwb"
    apidoc(
        [
            "--force",
            "-o",
            str(source),
            str(package),
            # apidoc resolves exclude patterns against the working directory,
            # and Read the Docs runs Sphinx from docs/source, so anchor them.
            str(package / "vendor" / "*"),
            str(package / "*" / "tests"),
            str(package / "*" / "tests" / "*"),
            str(package / "*" / "test_*.py"),
            str(package / "*.pyx"),
        ]
    )
    parser = create_parser()
    text = [".. Generated from pycwb.cli.main; do not edit.\n"]
    for name, _, _ in COMMANDS:
        command_parser = next(
            action.choices[name]
            for action in parser._actions
            if getattr(action, "choices", None) and name in action.choices
        )
        text.extend([name, "~" * len(name), "", ".. code-block:: text", ""])
        text.extend("   " + line for line in command_parser.format_help().splitlines())
        text.append("")
    (source / "_cli_help.rst.inc").write_text("\n".join(text))


def setup(app):
    """Register a serial build initialization hook."""
    app.connect("builder-inited", generate)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
