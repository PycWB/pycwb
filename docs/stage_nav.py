"""Pipeline stage navigator shown under the title of concept pages.

Usage::

    .. stage-nav:: search
       :current: likelihood

    .. stage-nav:: search
       :touches: conditioning events

       Optional one-line note explaining how the page relates to the track.

``:current:`` marks the stage this page documents. ``:touches:`` marks stages
that a cross-cutting page (injections, sky masks) acts on. Stage names, order,
and link targets are defined once in ``TRACKS`` so every page shows the same
sequence.
"""

from docutils import nodes
from docutils.parsers.rst import directives
from sphinx.addnodes import pending_xref
from sphinx.util.docutils import SphinxDirective

# (key, label, reference label). Every stage links to its own concept page;
# the track title links to the overview of the whole track.
TRACKS = {
    "search": {
        "title": "Search pipeline",
        "overview": "pipeline_lifecycle",
        "stages": [
            ("segments", "Segments", "job_control"),
            ("data", "Data", "data_ingestion"),
            ("conditioning", "Conditioning", "data_conditioning"),
            ("wdm", "WDM", "wdm_transform"),
            ("clusters", "Clusters", "clustering_algorithm"),
            ("likelihood", "Likelihood", "likelihood_guide"),
            ("events", "Events", "event_output"),
        ],
        "neighbor": ("Postproduction \u2192", "postproduction"),
    },
    "postproduction": {
        "title": "Postproduction",
        "overview": "postproduction",
        "stages": [
            ("background", "Background", "postproduction_background"),
            ("training", "Training set", "postproduction_trainingset"),
            ("ranking", "Ranking", "postproduction_xgboost"),
            ("efficiency", "Efficiency", "postproduction_efficiency"),
            ("report", "Report", "postproduction_report"),
        ],
        "neighbor": ("\u2190 Search pipeline", "pipeline_lifecycle"),
    },
}


class stage_nav(nodes.General, nodes.Element):
    """An HTML element with a fixed tag; children are ordinary docutils nodes."""


def _part(tag, classes, *children, **attrs):
    return stage_nav("", *children, tag=tag, classes=classes, attrs=attrs)


class StageNavDirective(SphinxDirective):
    required_arguments = 1
    has_content = True
    option_spec = {
        "current": directives.unchanged,
        "touches": directives.unchanged,
    }

    def _link(self, text, target, classes):
        xref = pending_xref(
            "",
            nodes.inline(text, text),
            refdomain="std",
            reftype="ref",
            reftarget=target,
            refexplicit=True,
            refwarn=True,
            refdoc=self.env.docname,
        )
        return nodes.inline("", "", xref, classes=classes)

    def run(self):
        name = self.arguments[0]
        if name not in TRACKS:
            raise self.error(f"unknown stage-nav track {name!r}; expected one of {sorted(TRACKS)}")
        track = TRACKS[name]
        keys = [key for key, _, _ in track["stages"]]

        current = self.options.get("current", "").strip() or None
        touches = self.options.get("touches", "").split()
        for key in ([current] if current else []) + touches:
            if key not in keys:
                raise self.error(f"unknown {name} stage {key!r}; expected one of {keys}")

        heading = [self._link(track["title"], track["overview"], ["stage-nav__track"])]
        if current:
            index = keys.index(current) + 1
            text = f"Stage {index} of {len(keys)}"
            heading.append(nodes.inline(text, text, classes=["stage-nav__count"]))
        neighbor_text, neighbor_target = track["neighbor"]
        header = _part(
            "div",
            ["stage-nav__header"],
            _part("span", ["stage-nav__title"], *heading),
            self._link(neighbor_text, neighbor_target, ["stage-nav__neighbor"]),
        )

        steps = _part("ol", ["stage-nav__steps"])
        upstream = current is not None
        for number, (key, label, target) in enumerate(track["stages"], start=1):
            classes = ["stage-nav__step"]
            attrs = {}
            marker = nodes.inline(str(number), str(number), classes=["stage-nav__marker"])
            if key == current:
                upstream = False
                classes.append("is-current")
                attrs["aria-current"] = "page"
                body = nodes.inline(label, label, classes=["stage-nav__label"])
            else:
                if upstream:
                    classes.append("is-upstream")
                if key in touches:
                    classes.append("is-touched")
                body = self._link(label, target, ["stage-nav__label"])
            steps += _part("li", classes, marker, body, **attrs)

        nav = _part("nav", ["stage-nav"], header, steps, **{"aria-label": f"{track['title']} stages"})
        if self.content:
            note = _part("div", ["stage-nav__note"])
            self.state.nested_parse(self.content, self.content_offset, note)
            nav += note
        return [nav]


def visit_stage_nav_html(self, node):
    self.body.append(self.starttag(node, node["tag"], "", **node["attrs"]))


def depart_stage_nav_html(self, node):
    self.body.append(f"</{node['tag']}>")


def skip_stage_nav(self, node):
    # The navigator is web chrome; other output formats omit it.
    raise nodes.SkipNode


def setup(app):
    app.add_node(
        stage_nav,
        html=(visit_stage_nav_html, depart_stage_nav_html),
        latex=(skip_stage_nav, None),
        text=(skip_stage_nav, None),
        man=(skip_stage_nav, None),
        texinfo=(skip_stage_nav, None),
    )
    app.add_directive("stage-nav", StageNavDirective)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
