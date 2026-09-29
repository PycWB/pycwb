"""Refresh browser-cached figures when their source contents change."""

from hashlib import sha256
from html import escape
from pathlib import Path

from sphinx.util.osutil import relative_uri


def version_image_urls(app, pagename, templatename, context, doctree):
    body = context.get("body", "")
    for source, filename in app.builder.images.items():
        uri = escape(relative_uri(
            app.builder.get_target_uri(pagename),
            f"{app.builder.imagedir}/{filename}",
        ), quote=True)
        attributes = [f'{name}="{uri}"' for name in ("src", "href")]
        if not any(attribute in body for attribute in attributes):
            continue
        # Read the source: Sphinx copies images to the output after writing HTML.
        path = Path(app.srcdir) / source
        version = sha256(path.read_bytes()).hexdigest()[:12]
        for attribute in attributes:
            body = body.replace(attribute, f'{attribute[:-1]}?v={version}"')
    context["body"] = body


def setup(app):
    app.connect("html-page-context", version_image_urls)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
