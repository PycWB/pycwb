"""Workflow DAG diagram generator for pycwb post-production pipelines.

Parses a YAML workflow configuration, discovers dependencies between
steps via the ``@action_spec`` decorator metadata (and fallback
heuristics), and renders the resulting DAG via multiple backends:

- **Mermaid** (``.mmd``) — always generated, zero dependencies, renders
  natively in VS Code, GitHub, and most Markdown viewers.
- **Graphviz DOT** (``.dot`` → ``.png``) — highest quality offline PNG,
  requires ``dot`` binary (``brew install graphviz`` / ``apt install graphviz``).
- **Mermaid.ink** (HTTP → ``.png``) — pip-only PNG fallback, no binary
  needed.  Uses the free mermaid.ink API.  Requires internet.
- **D2** (``.d2`` → ``.png``) — modern layout engine with cleaner DAGs,
  requires ``d2`` binary (``brew install d2`` / ``curl -fsSL https://d2lang.com/install.sh | sh``).

PNG generation cascade (first available wins): **dot** → **mermaid.ink** → **d2**.
``.mmd`` file is always generated regardless.

Usage
-----
>>> from pycwb.post_production.diagram import build_dag, render_diagram
>>> import yaml
>>> with open('workflow.yaml') as f:
...     wf = yaml.safe_load(f)
>>> dag = build_dag(wf)
>>> render_diagram(dag, 'workflow_diagram')  # → .mmd + .png

CLI (via post_process.py)
-------------------------
``pycwb post_process --diagram-only workflow.yaml``
    Generate diagram and exit (dry-run).

``pycwb post_process workflow.yaml``
    Generate diagram at start, then execute.
"""

from __future__ import annotations

import base64
import html as html_mod
import importlib
import json
import logging
import os
import subprocess
import textwrap
import zlib
from typing import Any, Optional

import yaml

from pycwb.post_production.action_spec import get_action_spec
from pycwb.utils.module import import_helper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short_name(action: str) -> str:
    """Return the function name from a dotted action path."""
    return action.rsplit('.', 1)[-1]


def _resolve_action_func(action: str):
    """Import and return the callable referenced by *action*.

    Parameters
    ----------
    action : str
        Dotted path, e.g. ``"postprocess.job_selector.select_jobs_by_livetime"``.

    Returns
    -------
    Callable or None
        The function, or ``None`` if the import fails.
    """
    func_name = action.split('.')[-1]
    module_name = '.'.join(action.split('.')[:-1])

    # Try multiple import strategies:
    strategies = [module_name]  # as-is
    if not module_name.startswith('pycwb'):
        strategies.append(f'pycwb.modules.{module_name}')  # shorthand → full path

    for mod_name in strategies:
        try:
            module = import_helper(mod_name, mod_name)
            return getattr(module, func_name)
        except Exception:
            continue

    logger.debug("Could not import action %s (tried %s)", action, strategies)
    return None


# ---------------------------------------------------------------------------
# DAG construction
# ---------------------------------------------------------------------------

def build_dag(workflow: dict) -> dict:
    """Build a DAG from a parsed YAML workflow dictionary.

    Parameters
    ----------
    workflow : dict
        Parsed YAML with ``'global'`` and ``'steps'`` keys.

    Returns
    -------
    dict
        ``{'nodes': [...], 'edges': [...]}`` where each node is
        ``{'id': str, 'label': str, 'description': str, 'action': str,
          'step_index': int}`` and each edge is
        ``{'from': str, 'to': str, 'label': str, 'via': str}``.
        ``via`` is one of ``'file'``, ``'alias'``, or ``'heuristic'``.
    """
    steps = workflow.get('steps', [])
    global_args = workflow.get('global', {})

    # ---- build nodes ----
    nodes: list[dict] = []
    for i, step in enumerate(steps):
        action = step.get('action', '')
        func = _resolve_action_func(action)
        spec = get_action_spec(func) if func else {}
        nodes.append({
            'id': f'step{i}',
            'label': _short_name(action),
            'description': spec.get('description', _short_name(action)),
            'action': action,
            'step_index': i,
            'output_alias': step.get('output_alias', None),
            'args': step.get('args', {}),
        })

    # ---- build known-output map: step_id → {param_name: arg_value} ----
    # Start with global keys (e.g. work_dir, search, nifo) as "always available"
    known_outputs: dict[str, dict[str, str]] = {}
    # Also track output_alias → step_id for alias-based matching
    alias_to_step: dict[str, str] = {}

    for node in nodes:
        sid = node['id']
        step_outputs: dict[str, str] = {}

        func = _resolve_action_func(node['action'])
        spec = get_action_spec(func) if func else {}

        for out_param in spec.get('outputs', []):
            val = node['args'].get(out_param)
            if val is not None:
                step_outputs[out_param] = str(val)

        # Also track output_alias
        alias = node.get('output_alias')
        if alias:
            alias_to_step[str(alias)] = sid

        known_outputs[sid] = step_outputs

    # ---- special handling for compute_livetime and other dict-spread steps ----
    # These steps spread their return dict into global_args. The dict keys
    # (e.g. 'livetime', 'livetime_days') become available to downstream steps.
    # We detect these by looking for steps that return known keys in kwargs
    # of downstream steps, even without explicit output_file matching.
    spread_keys = {'livetime', 'livetime_days', 'n_jobs_selected',
                   'n_jobs_total', 'livetime_selected', 'livetime_total',
                   'n_before', 'n_after', 'filtered_file', 'job_ids_file',
                   'auc', 'model_file'}

    # Augment known_outputs with spread keys from steps that have no
    # output_alias and return dicts
    for node in nodes:
        sid = node['id']
        func = _resolve_action_func(node['action'])
        spec = get_action_spec(func) if func else {}
        out_params = set(spec.get('outputs', []))
        # If step has no output_alias, its return dict is spread —
        # add the spread keys as implicit outputs
        if not node.get('output_alias') and not out_params:
            for key in spread_keys:
                # Only add if it's a reasonable guess — check if any
                # downstream step actually uses this key as an arg value
                known_outputs.setdefault(sid, {})[f'_spread_{key}'] = key

    # ---- build edges ----
    edges: list[dict] = []
    for i, target in enumerate(nodes):
        tid = target['id']
        target_args = target['args']

        # Check each earlier step as potential source
        for j in range(i):
            source = nodes[j]
            sid = source['id']
            src_outputs = known_outputs.get(sid, {})

            # --- file-based matching ---
            func_t = _resolve_action_func(target['action'])
            spec_t = get_action_spec(func_t) if func_t else {}
            for in_param in spec_t.get('inputs', []):
                in_val = target_args.get(in_param)
                if in_val is None:
                    continue
                in_val_str = str(in_val)
                # Match against source outputs
                for out_param, out_val in src_outputs.items():
                    if out_param.startswith('_spread_'):
                        # spread key — match by key name appearing in arg value
                        spread_key = out_val
                        if spread_key in in_val_str:
                            edges.append({
                                'from': sid, 'to': tid,
                                'label': f'[{spread_key}]',
                                'via': 'alias',
                            })
                            break
                    elif out_val == in_val_str:
                        edges.append({
                            'from': sid, 'to': tid,
                            'label': os.path.basename(in_val_str),
                            'via': 'file',
                        })
                        break

            # --- alias-based matching ---
            # Check if any target arg value matches a source output_alias name
            src_alias = source.get('output_alias')
            if src_alias:
                for arg_key, arg_val in target_args.items():
                    if str(arg_val) == str(src_alias):
                        edges.append({
                            'from': sid, 'to': tid,
                            'label': f'alias: {src_alias}',
                            'via': 'alias',
                        })
                        break

    return {'nodes': nodes, 'edges': edges}


# ---------------------------------------------------------------------------
# Mermaid renderer
# ---------------------------------------------------------------------------

def render_mermaid(dag: dict, title: str = "Workflow DAG") -> str:
    """Render *dag* as a Mermaid flowchart string.

    Parameters
    ----------
    dag : dict
        Output of :func:`build_dag`.
    title : str
        Diagram title.

    Returns
    -------
    str
        Mermaid ``graph TD`` markup (suitable for ``.mmd`` files).
    """
    lines = ['```mermaid', f'---', f'title: {title}', f'---', 'graph TD']

    # Class definitions for styling
    lines.append('    classDef source fill:#e1f5fe,stroke:#0288d1')
    lines.append('    classDef intermediate fill:#fff9c4,stroke:#fbc02d')
    lines.append('    classDef sink fill:#f3e5f5,stroke:#7b1fa2')
    lines.append('    classDef standalone fill:#e0e0e0,stroke:#9e9e9e')

    # Determine node roles
    has_incoming: set[str] = set()
    has_outgoing: set[str] = set()
    for edge in dag['edges']:
        has_outgoing.add(edge['from'])
        has_incoming.add(edge['to'])

    # Render nodes
    for node in dag['nodes']:
        nid = node['id']
        idx = node['step_index'] + 1
        label = node['label']
        desc = node['description']
        # Escape quotes in description
        desc_escaped = desc.replace('"', '\\"')
        # Truncate long descriptions
        if len(desc_escaped) > 60:
            desc_escaped = desc_escaped[:57] + '...'
        node_text = f'<b>{idx}. {label}</b><br/>{desc_escaped}'
        lines.append(f'    {nid}["{node_text}"]')

        # Assign class
        if nid not in has_incoming and nid not in has_outgoing:
            lines.append(f'    class {nid} standalone')
        elif nid not in has_incoming:
            lines.append(f'    class {nid} source')
        elif nid not in has_outgoing:
            lines.append(f'    class {nid} sink')
        else:
            lines.append(f'    class {nid} intermediate')

    # Render edges
    for edge in dag['edges']:
        src = edge['from']
        dst = edge['to']
        lbl = edge['label'].replace('"', '\\"')
        if len(lbl) > 40:
            lbl = lbl[:37] + '...'
        lines.append(f'    {src} -->|"{lbl}"| {dst}')

    lines.append('```')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Graphviz DOT renderer
# ---------------------------------------------------------------------------

def render_dot(dag: dict, title: str = "Workflow DAG") -> str:
    """Render *dag* as a Graphviz DOT string.

    Parameters
    ----------
    dag : dict
        Output of :func:`build_dag`.
    title : str
        Diagram title / graph label.

    Returns
    -------
    str
        Graphviz DOT markup.
    """
    lines = [
        'digraph workflow {',
        f'    label="{title}";',
        '    labelloc=t;',
        '    fontsize=16;',
        '    rankdir=LR;',
        '    newrank=true;',
        '    splines=ortho;',
        '    ranksep=0.5;',
        '    nodesep=0.3;',
        '    pad=0.3;',
        '    dpi=150;',
        '    node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.12,0.06"];',
        '    edge [fontname="Helvetica", fontsize=8, arrowsize=0.7];',
        '',
    ]

    # Node styles
    has_incoming: set[str] = set()
    has_outgoing: set[str] = set()
    for edge in dag['edges']:
        has_outgoing.add(edge['from'])
        has_incoming.add(edge['to'])

    for node in dag['nodes']:
        nid = node['id']
        idx = node['step_index'] + 1
        label = node['label']
        desc = node['description']
        desc_escaped = desc.replace('"', '\\"').replace('\n', '\\n')
        # Shorter truncation for more compact nodes
        if len(desc_escaped) > 45:
            desc_escaped = desc_escaped[:42] + '...'
        node_label = f'{idx}. {label}\\n{desc_escaped}'

        # Color by role
        if nid not in has_incoming and nid not in has_outgoing:
            color = '#e0e0e0'  # standalone
            fontcolor = '#555555'
        elif nid not in has_incoming:
            color = '#bbdefb'  # source — lighter blue
            fontcolor = '#000000'
        elif nid not in has_outgoing:
            color = '#e1bee7'  # sink — lighter purple
            fontcolor = '#000000'
        else:
            color = '#fff9c4'  # intermediate — lighter yellow
            fontcolor = '#000000'

        lines.append(
            f'    {nid} [label="{node_label}", fillcolor="{color}", '
            f'fontcolor="{fontcolor}"];'
        )

    lines.append('')

    # Edges
    for edge in dag['edges']:
        src = edge['from']
        dst = edge['to']
        lbl = edge['label'].replace('"', '\\"')
        if len(lbl) > 30:
            lbl = lbl[:27] + '...'
        # Dashed for alias, solid for file
        style = 'dashed' if edge.get('via') == 'alias' else 'solid'
        lines.append(f'    {src} -> {dst} [label="{lbl}", style={style}];')

    lines.append('}')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# D2 renderer (https://d2lang.com) — modern alternative to Graphviz
# ---------------------------------------------------------------------------

def render_d2(dag: dict, title: str = "Workflow DAG") -> str:
    """Render *dag* as a D2 markup string.

    D2 is a modern declarative diagram language with the TALA layout
    engine.  Produces cleaner, less-cluttered DAGs than Graphviz.
    Requires ``d2`` binary to render to PNG (``brew install d2``).

    Parameters
    ----------
    dag : dict
        Output of :func:`build_dag`.
    title : str
        Diagram title.

    Returns
    -------
    str
        D2 markup.
    """
    lines = [
        f'# {title}',
        '',
        'direction: right',
        '',
    ]

    # Determine node roles for coloring
    has_incoming: set[str] = set()
    has_outgoing: set[str] = set()
    for edge in dag['edges']:
        has_outgoing.add(edge['from'])
        has_incoming.add(edge['to'])

    for node in dag['nodes']:
        nid = node['id']
        idx = node['step_index'] + 1
        label = node['label']
        desc = node['description']
        # Escape quotes
        desc_escaped = desc.replace('"', "'")
        if len(desc_escaped) > 50:
            desc_escaped = desc_escaped[:47] + '...'

        # Color by role
        if nid not in has_incoming and nid not in has_outgoing:
            fill = '#e0e0e0'
        elif nid not in has_incoming:
            fill = '#bbdefb'
        elif nid not in has_outgoing:
            fill = '#e1bee7'
        else:
            fill = '#fff9c4'

        lines.append(f'{nid}: "{idx}. {label}\\n{desc_escaped}" {{')
        lines.append(f'  style.fill: "{fill}"')
        lines.append('}')

    lines.append('')

    # Edges
    for edge in dag['edges']:
        src = edge['from']
        dst = edge['to']
        lbl = edge['label'].replace('"', "'")
        if len(lbl) > 35:
            lbl = lbl[:32] + '...'
        style = 'stroke-dash: 6' if edge.get('via') == 'alias' else ''
        if style:
            lines.append(f'{src} -> {dst}: "{lbl}" {{ {style} }}')
        else:
            lines.append(f'{src} -> {dst}: "{lbl}"')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Interactive HTML renderer (vis.js) — zero dependencies, browser-native
# ---------------------------------------------------------------------------

def render_html(dag: dict, title: str = "Workflow DAG") -> str:
    """Render *dag* as a self-contained interactive HTML page using vis.js.

    Features:
    - Hierarchical left-to-right layout (Sugiyama algorithm via vis.js)
    - Pan &amp; zoom (mouse wheel / touch)
    - Click a node to highlight its upstream &amp; downstream connections
    - Hover over edges to see the connecting file name
    - Color-coded nodes: blue=source, yellow=intermediate, purple=sink
    - Responsive — fills the browser window
    - Single file, works offline after first CDN load

    Parameters
    ----------
    dag : dict
        Output of :func:`build_dag`.
    title : str
        Page title.

    Returns
    -------
    str
        Self-contained HTML document.
    """
    # Determine node roles
    has_incoming: set[str] = set()
    has_outgoing: set[str] = set()
    for edge in dag['edges']:
        has_outgoing.add(edge['from'])
        has_incoming.add(edge['to'])

    # Build vis.js node data
    vis_nodes: list[dict] = []
    for node in dag['nodes']:
        nid = node['id']
        idx = node['step_index'] + 1
        label = node['label']
        desc = node['description']
        action = node['action']

        # Color by role
        if nid not in has_incoming and nid not in has_outgoing:
            color = '#e0e0e0'
            border = '#9e9e9e'
        elif nid not in has_incoming:
            color = '#bbdefb'
            border = '#0288d1'
        elif nid not in has_outgoing:
            color = '#e1bee7'
            border = '#7b1fa2'
        else:
            color = '#fff9c4'
            border = '#fbc02d'

        vis_nodes.append({
            'id': nid,
            'label': f'{idx}. {label}',
            'title': f'<b>{idx}. {label}</b><br>{html_mod.escape(desc)}<br><code>{html_mod.escape(action)}</code>',
            'color': {'background': color, 'border': border},
            'font': {'size': 13, 'face': 'Helvetica, Arial, sans-serif'},
            'shape': 'box',
            'borderWidth': 2,
            'margin': {'top': 8, 'bottom': 8, 'left': 12, 'right': 12},
        })

    # Build vis.js edge data
    vis_edges: list[dict] = []
    for edge in dag['edges']:
        is_alias = edge.get('via') == 'alias'
        vis_edges.append({
            'from': edge['from'],
            'to': edge['to'],
            'label': edge['label'],
            'title': f'<b>{html_mod.escape(edge["label"])}</b><br>via: {edge.get("via", "file")}',
            'arrows': 'to',
            'dashes': is_alias,
            'font': {'size': 9, 'face': 'Helvetica, Arial, sans-serif',
                     'color': '#666666'},
            'color': {'color': '#999999' if is_alias else '#555555'},
            'smooth': {'type': 'cubicBezier', 'roundness': 0.3},
        })

    nodes_json = json.dumps(vis_nodes, indent=2)
    edges_json = json.dumps(vis_edges, indent=2)
    title_esc = html_mod.escape(title)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title_esc}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js">
</script>
<link rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: Helvetica, Arial, sans-serif; overflow: hidden; }}
  #header {{
    background: #263238; color: #ffffff; padding: 10px 20px;
    display: flex; align-items: center; justify-content: space-between;
    font-size: 14px;
  }}
  #header h1 {{ font-size: 16px; font-weight: 500; margin: 0; }}
  #header .legend {{
    display: flex; gap: 14px; font-size: 12px; align-items: center;
  }}
  #header .legend span {{ display: flex; align-items: center; gap: 5px; }}
  #header .legend .dot {{
    width: 12px; height: 12px; border-radius: 3px; display: inline-block;
  }}
  #mynetwork {{ width: 100vw; height: calc(100vh - 44px); background: #fafafa; }}
  .vis-network:focus {{ outline: none; }}
  .vis-tooltip {{ max-width: 320px; }}
</style>
</head>
<body>
<div id="header">
  <h1>{title_esc}</h1>
  <div class="legend">
    <span><span class="dot" style="background:#bbdefb;border:2px solid #0288d1"></span> Source</span>
    <span><span class="dot" style="background:#fff9c4;border:2px solid #fbc02d"></span> Intermediate</span>
    <span><span class="dot" style="background:#e1bee7;border:2px solid #7b1fa2"></span> Sink</span>
    <span style="color:#888">| Click node → highlight | Scroll → zoom</span>
  </div>
</div>
<div id="mynetwork"></div>

<script>
  var nodes = new vis.DataSet({nodes_json});
  var edges = new vis.DataSet({edges_json});

  var container = document.getElementById('mynetwork');
  var data = {{ nodes: nodes, edges: edges }};
  var options = {{
    layout: {{
      hierarchical: {{
        enabled: true,
        direction: 'LR',
        sortMethod: 'directed',
        levelSeparation: 220,
        nodeSpacing: 160,
        treeSpacing: 100,
        blockShifting: true,
        edgeMinimization: true,
      }},
    }},
    physics: {{ enabled: false }},
    interaction: {{
      hover: true,
      tooltipDelay: 200,
      navigationButtons: true,
      keyboard: true,
    }},
    edges: {{
      smooth: {{ type: 'cubicBezier', roundness: 0.3 }},
      font: {{ align: 'top', size: 9 }},
    }},
    nodes: {{
      shape: 'box',
      borderWidth: 2,
      shadow: {{ enabled: true, size: 4, x: 1, y: 1 }},
    }},
  }};

  var network = new vis.Network(container, data, options);

  // Click-to-highlight: select node → dim unrelated nodes/edges
  var highlighted = null;
  network.on('click', function(params) {{
    if (highlighted === params.nodes[0]) {{
      // Deselect
      highlighted = null;
      nodes.forEach(function(n) {{ nodes.update({{id: n.id, opacity: 1.0}}); }});
      edges.forEach(function(e) {{ edges.update({{id: e.id, opacity: 1.0, color: e.color}}); }});
      return;
    }}
    highlighted = params.nodes[0];
    if (!highlighted) return;

    var connected = new Set([highlighted]);
    edges.forEach(function(e) {{
      if (e.from === highlighted) connected.add(e.to);
      if (e.to === highlighted) connected.add(e.from);
    }});

    nodes.forEach(function(n) {{
      nodes.update({{id: n.id, opacity: connected.has(n.id) ? 1.0 : 0.15}});
    }});
    edges.forEach(function(e) {{
      var active = e.from === highlighted || e.to === highlighted;
      edges.update({{
        id: e.id,
        opacity: active ? 1.0 : 0.1,
        color: active ? e.color : {{color: '#dddddd'}},
      }});
    }});
  }});

  // Fit on load
  network.once('stabilized', function() {{
    network.fit({{ animation: {{ duration: 500 }} }});
  }});
</script>
</body>
</html>'''

def _render_mermaid_png(mermaid_text: str, png_path: str) -> bool:
    """Render Mermaid markup to PNG via the mermaid.ink HTTP API.

    Encodes the Mermaid markup (without the `` ```mermaid `` wrapper)
    as base64 → deflate → base64url and fetches the rendered image
    from ``https://mermaid.ink/img/<encoded>``.

    Parameters
    ----------
    mermaid_text : str
        Mermaid markup including the `` ```mermaid `` fence.
    png_path : str
        Destination path for the PNG file.

    Returns
    -------
    bool
        True if the PNG was successfully saved.
    """
    try:
        import urllib.request
    except ImportError:
        logger.debug("urllib.request not available — skipping mermaid.ink")
        return False

    # Strip the ```mermaid fence and frontmatter for encoding
    core = mermaid_text
    # Remove ```mermaid wrapper
    if core.startswith('```mermaid'):
        core = core[len('```mermaid'):]
    if core.endswith('```'):
        core = core[:-len('```')]
    # Remove YAML frontmatter (--- ... ---)
    core = core.strip()
    if core.startswith('---'):
        # Find second ---
        idx = core.find('---', 3)
        if idx != -1:
            core = core[idx + 3:].strip()

    # mermaid.ink encoding: pako (deflate) → base64url
    # Python equivalent: raw bytes → zlib → base64url
    raw_bytes = core.encode('utf-8')
    compressed = zlib.compress(raw_bytes, level=9)
    encoded = base64.urlsafe_b64encode(compressed).rstrip(b'=').decode('ascii')

    url = f'https://mermaid.ink/img/{encoded}'
    logger.debug('Fetching mermaid.ink: %s...', url[:80])

    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'pycwb-diagram/1.0',
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            png_data = resp.read()
        os.makedirs(os.path.dirname(png_path) or '.', exist_ok=True)
        with open(png_path, 'wb') as f:
            f.write(png_data)
        logger.info('Mermaid PNG (via mermaid.ink) written to %s', png_path)
        return True
    except Exception as exc:
        logger.debug('mermaid.ink request failed: %s', exc)
        return False


# ---------------------------------------------------------------------------
# File output (multi-backend cascade)
# ---------------------------------------------------------------------------

def render_diagram(
    dag: dict,
    output_prefix: str,
    title: str = "Workflow DAG",
    generate_png: bool = True,
) -> dict[str, Optional[str]]:
    """Write Mermaid (``.mmd``), D2, DOT, and optionally PNG diagram files.

    PNG generation uses a cascade — the first available backend wins:

    1. **dot** (Graphviz) — best offline quality
    2. **mermaid.ink** — HTTP API, pip-only, needs internet
    3. **d2** — modern layout, needs ``d2`` binary

    ``.mmd`` is always generated regardless.

    Parameters
    ----------
    dag : dict
        Output of :func:`build_dag`.
    output_prefix : str
        Path prefix for output files (e.g. ``'workflow_diagram'``
        produces ``workflow_diagram.mmd``, ``workflow_diagram.png``, etc.).
    title : str
        Diagram title.
    generate_png : bool
        If True, attempt PNG generation via the cascade above.

    Returns
    -------
    dict
        ``{'mmd': path, 'png': path or None, 'png_method': str, 'dot': path, 'd2': path}``
    """
    result: dict[str, Optional[str]] = {
        'mmd': None, 'png': None, 'png_method': None,
        'dot': None, 'd2': None, 'html': None,
    }

    # --- Mermaid (.mmd) — always generated ---
    mmd_path = f'{output_prefix}.mmd'
    mermaid_text = render_mermaid(dag, title)
    with open(mmd_path, 'w') as f:
        f.write(mermaid_text)
    result['mmd'] = mmd_path
    logger.info('Mermaid diagram written to %s', mmd_path)

    # --- Interactive HTML (.html) — always generated, zero deps ---
    html_path = f'{output_prefix}.html'
    html_text = render_html(dag, title)
    with open(html_path, 'w') as f:
        f.write(html_text)
    result['html'] = html_path
    logger.info('Interactive HTML diagram written to %s', html_path)

    # --- D2 (.d2) — always generated (text file, no binary needed) ---
    d2_path = f'{output_prefix}.d2'
    d2_text = render_d2(dag, title)
    with open(d2_path, 'w') as f:
        f.write(d2_text)
    result['d2'] = d2_path
    logger.info('D2 diagram written to %s', d2_path)

    # --- Graphviz DOT — always generated (text file, no binary needed) ---
    dot_path = f'{output_prefix}.dot'
    dot_text = render_dot(dag, title)
    with open(dot_path, 'w') as f:
        f.write(dot_text)
    result['dot'] = dot_path
    logger.info('DOT diagram written to %s', dot_path)

    # --- PNG (cascade: dot → mermaid.ink → d2) ---
    if generate_png:
        png_path = f'{output_prefix}.png'
        png_ok = False

        # Priority 1: Graphviz dot
        if not png_ok:
            png_ok = _render_dot_png(dot_path, png_path)
            if png_ok:
                result['png_method'] = 'graphviz'

        # Priority 2: mermaid.ink HTTP API
        if not png_ok:
            png_ok = _render_mermaid_png(mermaid_text, png_path)
            if png_ok:
                result['png_method'] = 'mermaid.ink'

        # Priority 3: D2
        if not png_ok:
            png_ok = _render_d2_png(d2_path, png_path)
            if png_ok:
                result['png_method'] = 'd2'

        if png_ok:
            result['png'] = png_path
        else:
            logger.warning(
                'No PNG backend available. Install one of:\n'
                '  brew install graphviz   (dot — best quality)\n'
                '  brew install d2         (d2 — modern layout)\n'
                '  or use online with internet access (mermaid.ink fallback)'
            )

    return result


def _render_dot_png(dot_path: str, png_path: str) -> bool:
    """Render DOT → PNG via the ``dot`` binary.  Returns True on success."""
    try:
        subprocess.run(
            ['dot', '-Tpng', dot_path, '-o', png_path],
            check=True, capture_output=True, text=True,
        )
        logger.info('PNG diagram (graphviz) written to %s', png_path)
        return True
    except FileNotFoundError:
        logger.debug('Graphviz "dot" not found')
        return False
    except subprocess.CalledProcessError as exc:
        logger.debug('Graphviz rendering failed: %s', exc.stderr.strip())
        return False


def _render_d2_png(d2_path: str, png_path: str) -> bool:
    """Render D2 → PNG via the ``d2`` binary.  Returns True on success."""
    try:
        subprocess.run(
            ['d2', d2_path, png_path],
            check=True, capture_output=True, text=True,
        )
        logger.info('PNG diagram (d2) written to %s', png_path)
        return True
    except FileNotFoundError:
        logger.debug('D2 "d2" binary not found')
        return False
    except subprocess.CalledProcessError as exc:
        logger.debug('D2 rendering failed: %s', exc.stderr.strip())
        return False


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------

def generate_workflow_diagram(
    workflow_file: str,
    output_prefix: Optional[str] = None,
    generate_png: bool = True,
) -> dict:
    """Parse a workflow YAML, build the DAG, and write diagram files.

    This is the main entry point called by the CLI and by
    :func:`pycwb.post_production.workflow.run_workflow`.

    Parameters
    ----------
    workflow_file : str
        Path to the YAML workflow file.
    output_prefix : str, optional
        Prefix for output files.  Defaults to
        ``<workflow_dir>/workflow_diagram``.
    generate_png : bool
        Attempt PNG generation via Graphviz.

    Returns
    -------
    dict
        ``{'mmd': path, 'png': path or None, 'dot': path, 'dag': dag_dict}``
    """
    with open(workflow_file, 'r') as f:
        workflow = yaml.safe_load(f)

    if output_prefix is None:
        work_dir = workflow.get('global', {}).get('work_dir', '.')
        output_prefix = os.path.join(work_dir, 'workflow_diagram')

    dag = build_dag(workflow)
    files = render_diagram(dag, output_prefix,
                           title=f'Workflow: {os.path.basename(workflow_file)}',
                           generate_png=generate_png)
    files['dag'] = dag
    return files
