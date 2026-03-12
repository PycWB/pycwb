"""
Interactive D3.js / HTML visualization for FragmentCluster objects.

Usage::

    from pycwb.modules.plot.fragment_cluster_viz import plot_fragment_clusters

    plot_fragment_clusters(
        [fc1, fc2],
        labels=["Signal candidates", "Noise glitches"],
        output_path="clusters.html",
        title="cWB Fragment Clusters",
    )

Each ``FragmentCluster`` becomes a filterable layer in the sidebar.  Within a
layer every ``Cluster`` is rendered as a distinct colour.  Pixels are drawn as
axis-aligned rectangles whose time-width equals ``1/rate`` seconds and whose
frequency-height equals ``rate/2`` Hz — the natural tile size of the WDM
decomposition — so pixels at different resolutions automatically appear with
different sizes.

Performance notes
-----------------
* Pixel geometry is serialised once in Python as compact JSON arrays.
* The browser renders via ``<canvas>`` using colour-batched ``Path2D``/
  ``beginPath`` calls rather than individual SVG elements.
* Viewport culling skips pixels outside the current view on every frame.
* ``requestAnimationFrame`` coalesces rapid zoom/pan redraws.
"""

import json
import os
from typing import Optional


# ---------------------------------------------------------------------------
# Python-side data preparation
# ---------------------------------------------------------------------------

def _serialize(fragment_clusters, labels):
    """
    Convert FragmentCluster list to a JSON-serialisable list.

    Each pixel is stored as a 6-element array for compactness:
        [x_left, y_bottom, width, height, is_core, likelihood]

    Time values are seconds (relative to the data segment origin, i.e. the
    raw WDM time index converted to seconds).  Frequency values are in Hz.
    """
    if labels is None:
        labels = [f"FragmentCluster {i}" for i in range(len(fragment_clusters))]
    else:
        labels = list(labels)
        while len(labels) < len(fragment_clusters):
            labels.append(f"FragmentCluster {len(labels)}")

    result = []
    global_cluster_id = 0

    for fc_idx, (fc, label) in enumerate(zip(fragment_clusters, labels)):
        fc_clusters = []
        for c_idx, cluster in enumerate(fc.clusters):
            pixels = []
            for pixel in cluster.pixels:
                dt = 1.0 / pixel.rate          # pixel time width (s)
                df = pixel.rate / 2.0          # pixel frequency height (Hz)
                # Left edge: consistent with Cluster.start_time formula
                x = round(int(pixel.time / pixel.layers) * dt, 7)
                # Bottom edge: consistent with Cluster.low_frequency formula
                y = round((pixel.frequency - 0.5) * df, 4)
                pixels.append([
                    x,
                    y,
                    round(dt, 7),
                    round(df, 4),
                    1 if pixel.core else 0,
                    round(float(pixel.likelihood), 4),
                ])
            fc_clusters.append({
                "id": global_cluster_id,
                "local_id": c_idx,
                "pixels": pixels,
            })
            global_cluster_id += 1

        result.append({
            "id": fc_idx,
            "label": str(label),
            "clusters": fc_clusters,
        })

    return result


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title><<<TITLE_RAW>>></title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden;font-family:'Segoe UI',system-ui,sans-serif;
  font-size:13px;background:#111827;color:#d1d5db}
#app{display:flex;height:100vh}

/* ---- sidebar ---- */
#sidebar{
  width:270px;min-width:200px;
  background:#1f2937;
  border-right:1px solid #374151;
  display:flex;flex-direction:column;
}
#sidebar-header{padding:10px 12px 6px;border-bottom:1px solid #374151}
#sidebar-header h3{font-size:13px;font-weight:600;color:#f9fafb}
#sidebar-controls{display:flex;gap:6px;margin-top:6px}
#fc-list{flex:1;overflow-y:auto;padding:6px 0}

/* ---- main ---- */
#main{flex:1;display:flex;flex-direction:column;min-width:0;overflow:hidden}
#info-bar{
  display:flex;align-items:center;gap:12px;
  padding:5px 12px;border-bottom:1px solid #374151;
  background:#1f2937;font-size:11px;color:#9ca3af;
}
#plot-title{font-weight:600;font-size:13px;color:#f9fafb;flex:1}
#info-counts{}
#info-zoom{}
#chart-wrapper{flex:1;position:relative;overflow:hidden;background:#030712}
#pixel-canvas{position:absolute;top:0;left:0;display:block}
#axis-svg{position:absolute;top:0;left:0}

/* ---- axes (D3) ---- */
.axis path,.axis line{stroke:#4b5563}
.axis text{fill:#9ca3af;font-size:10px}
.x-grid line,.y-grid line{stroke:#1f2937;stroke-dasharray:3,3}
.axis-label{fill:#6b7280;font-size:11px}

/* ---- tooltip ---- */
#tooltip{
  position:absolute;display:none;
  background:rgba(17,24,39,0.95);border:1px solid #4b5563;
  border-radius:6px;padding:8px 10px;font-size:11px;
  pointer-events:none;line-height:1.7;color:#d1d5db;
  max-width:220px;z-index:100;
}
#tooltip b{color:#f9fafb;font-size:12px}

/* ---- sidebar items ---- */
.fc-item{
  display:flex;align-items:flex-start;gap:7px;
  padding:6px 12px;border-bottom:1px solid #111827;cursor:pointer;
  transition:background 0.1s;
}
.fc-item:hover{background:#2d3748}
.fc-item.fc-hidden .fc-label,.fc-item.fc-hidden .fc-count{opacity:0.4}
.fc-swatches{
  display:flex;flex-direction:column;align-items:center;
  gap:2px;margin-top:2px;flex-shrink:0;
}
.fc-swatch{width:12px;height:12px;border-radius:2px;flex-shrink:0}
.fc-text{display:flex;flex-direction:column;gap:1px;min-width:0}
.fc-label{font-size:12px;color:#e5e7eb;word-break:break-word}
.fc-count{font-size:10px;color:#6b7280}
input[type=checkbox]{
  margin-top:2px;flex-shrink:0;
  accent-color:#3b82f6;cursor:pointer;width:14px;height:14px;
}

/* ---- buttons ---- */
button{
  background:#374151;color:#d1d5db;border:1px solid #4b5563;
  border-radius:4px;padding:3px 9px;cursor:pointer;font-size:11px;
  transition:background 0.1s;
}
button:hover{background:#4b5563}
#btn-reset{margin-left:auto;font-size:11px}
</style>
</head>
<body>
<div id="app">

  <!-- Sidebar -->
  <div id="sidebar">
    <div id="sidebar-header">
      <h3>Fragment Clusters</h3>
      <div id="sidebar-controls">
        <button id="btn-all">Show all</button>
        <button id="btn-none">Hide all</button>
      </div>
    </div>
    <div id="fc-list"></div>
  </div>

  <!-- Main chart area -->
  <div id="main">
    <div id="info-bar">
      <span id="plot-title"></span>
      <span id="info-counts"></span>
      <span id="info-zoom"></span>
      <button id="btn-reset">Reset view</button>
    </div>
    <div id="chart-wrapper">
      <canvas id="pixel-canvas"></canvas>
      <svg id="axis-svg"></svg>
      <div id="tooltip"></div>
    </div>
  </div>

</div>

<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<script>
// ============================================================
//  Data injected by Python
// ============================================================
const DATA  = <<<DATA>>>;
const TITLE = <<<TITLE_JSON>>>;

// ============================================================
//  Layout constants
// ============================================================
const M = { top: 20, right: 20, bottom: 48, left: 72 };

// ============================================================
//  Global state
// ============================================================
const state = {
  hiddenFCs : new Set(),
  transform : d3.zoomIdentity,
  rafId     : null,
};

// DOM refs assigned in init()
let canvas, ctx, svg, wrapper;
let xScale, yScale, xAxisG, yAxisG;
let zoomBehavior;
let W, H;

// ============================================================
//  Colour assignment
//  Each FC gets its own hue band; clusters within a FC cycle
//  lightness from 38 % → 68 %.  Golden-angle hue spacing gives
//  maximum perceptual separation between FCs.
// ============================================================
function assignColors() {
  const numFCs = DATA.length;
  const goldenAngle = 137.508;
  DATA.forEach((fc, fi) => {
    const hue  = Math.round((fi * goldenAngle) % 360);
    const n    = fc.clusters.length;
    fc.baseHue = hue;
    fc.clusters.forEach((cl, ci) => {
      const t = n > 1 ? ci / (n - 1) : 0.5;
      // Slight hue walk (±25°) so clusters are distinguishable even at same FC
      const h = ((hue + (t - 0.5) * 50) + 360) % 360;
      const l = Math.round(38 + t * 30);
      cl.color = `hsl(${Math.round(h)},72%,${l}%)`;
    });
  });
}

// ============================================================
//  Compute global data extents (called once)
// ============================================================
function computeBounds() {
  let xMin = Infinity, xMax = -Infinity;
  let yMin = Infinity, yMax = -Infinity;
  let total = 0;
  DATA.forEach(fc => {
    fc.clusters.forEach(cl => {
      // pixel = [x, y, w, h, core, like]
      cl.pixels.forEach(p => {
        if (p[0]        < xMin) xMin = p[0];
        if (p[0] + p[2] > xMax) xMax = p[0] + p[2];
        if (p[1]        < yMin) yMin = p[1];
        if (p[1] + p[3] > yMax) yMax = p[1] + p[3];
      });
      total += cl.pixels.length;
    });
  });
  return { xMin, xMax, yMin, yMax, total };
}

// ============================================================
//  Canvas rendering  —  colour-batched for performance
// ============================================================
function render() {
  state.rafId = null;

  const curX = state.transform.rescaleX(xScale);

  ctx.clearRect(0, 0, W, H);

  // Dark chart background
  ctx.fillStyle = '#030712';
  ctx.fillRect(M.left, M.top, W - M.left - M.right, H - M.top - M.bottom);

  // Clip to chart area
  ctx.save();
  ctx.beginPath();
  ctx.rect(M.left, M.top, W - M.left - M.right, H - M.top - M.bottom);
  ctx.clip();

  // Viewport bounds for culling
  const vxMin = curX.invert(M.left);
  const vxMax = curX.invert(W - M.right);
  const vyMin = yScale.invert(H - M.bottom);
  const vyMax = yScale.invert(M.top);

  // Collect pixels into per-colour buckets
  const buckets = new Map(); // color -> pixel[]
  let visible = 0;

  DATA.forEach(fc => {
    if (state.hiddenFCs.has(fc.id)) return;
    fc.clusters.forEach(cl => {
      let bucket = buckets.get(cl.color);
      if (!bucket) { bucket = []; buckets.set(cl.color, bucket); }
      cl.pixels.forEach(p => {
        // Viewport cull
        if (p[0] + p[2] <= vxMin || p[0] >= vxMax) return;
        if (p[1] + p[3] <= vyMin || p[1] >= vyMax) return;
        bucket.push(p);
        visible++;
      });
    });
  });

  // Draw each colour group as a single path (fast batch fill)
  buckets.forEach((pixels, color) => {
    ctx.fillStyle = color;
    ctx.beginPath();
    pixels.forEach(p => {
      const px = curX(p[0]);
      const pw = Math.max(curX(p[0] + p[2]) - px, 0.5);
      const py = yScale(p[1] + p[3]);
      const ph = Math.max(yScale(p[1]) - py, 0.5);
      ctx.rect(px, py, pw, ph);
    });
    ctx.fill();
  });

  ctx.restore();

  // Update D3 axes
  const xTicks = Math.max(4, Math.floor((W - M.left - M.right) / 80));
  const yTicks = Math.max(4, Math.floor((H - M.top  - M.bottom) / 50));
  xAxisG.call(d3.axisBottom(curX).ticks(xTicks));
  yAxisG.call(d3.axisLeft(yScale).ticks(yTicks));

  // Info bar
  document.getElementById('info-counts').textContent =
    `${visible.toLocaleString()} pixels visible`;
  document.getElementById('info-zoom').textContent =
    `Zoom ×${state.transform.k.toFixed(2)}`;
}

function scheduleRender() {
  if (state.rafId) cancelAnimationFrame(state.rafId);
  state.rafId = requestAnimationFrame(render);
}

// ============================================================
//  Sidebar
// ============================================================
function buildSidebar() {
  const list = document.getElementById('fc-list');
  list.innerHTML = '';

  DATA.forEach(fc => {
    const totalPx = fc.clusters.reduce((s, c) => s + c.pixels.length, 0);

    const item = document.createElement('div');
    item.className = 'fc-item';
    item.id = `fc-item-${fc.id}`;

    // Colour swatches (up to 5)
    const swatchWrap = document.createElement('div');
    swatchWrap.className = 'fc-swatches';
    const maxSw = Math.min(fc.clusters.length, 5);
    for (let i = 0; i < maxSw; i++) {
      const sw = document.createElement('div');
      sw.className = 'fc-swatch';
      sw.style.background = fc.clusters[i].color;
      swatchWrap.appendChild(sw);
    }

    // Checkbox
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = true;
    cb.id = `fc-cb-${fc.id}`;
    cb.addEventListener('change', e => {
      if (e.target.checked) state.hiddenFCs.delete(fc.id);
      else                  state.hiddenFCs.add(fc.id);
      item.classList.toggle('fc-hidden', !e.target.checked);
      scheduleRender();
    });

    // Text block
    const textWrap = document.createElement('label');
    textWrap.htmlFor = `fc-cb-${fc.id}`;
    textWrap.className = 'fc-text';
    textWrap.style.cursor = 'pointer';

    const labelEl = document.createElement('span');
    labelEl.className = 'fc-label';
    labelEl.textContent = fc.label;

    const countEl = document.createElement('span');
    countEl.className = 'fc-count';
    countEl.textContent =
      `${fc.clusters.length} cluster${fc.clusters.length !== 1 ? 's' : ''}, ` +
      `${totalPx.toLocaleString()} px`;

    textWrap.appendChild(labelEl);
    textWrap.appendChild(countEl);

    item.appendChild(swatchWrap);
    item.appendChild(cb);
    item.appendChild(textWrap);
    list.appendChild(item);
  });

  document.getElementById('btn-all').addEventListener('click', () => {
    DATA.forEach(fc => {
      state.hiddenFCs.delete(fc.id);
      document.getElementById(`fc-cb-${fc.id}`).checked = true;
      document.getElementById(`fc-item-${fc.id}`).classList.remove('fc-hidden');
    });
    scheduleRender();
  });

  document.getElementById('btn-none').addEventListener('click', () => {
    DATA.forEach(fc => {
      state.hiddenFCs.add(fc.id);
      document.getElementById(`fc-cb-${fc.id}`).checked = false;
      document.getElementById(`fc-item-${fc.id}`).classList.add('fc-hidden');
    });
    scheduleRender();
  });
}

// ============================================================
//  Tooltip  (debounced pixel hit-test)
// ============================================================
let _ttTimer = null;

function onMouseMove(event) {
  if (_ttTimer) clearTimeout(_ttTimer);
  _ttTimer = setTimeout(() => {
    const curX = state.transform.rescaleX(xScale);
    const [mx, my] = d3.pointer(event);
    const tooltip = document.getElementById('tooltip');

    // Must be inside chart area
    if (mx < M.left || mx > W - M.right || my < M.top || my > H - M.bottom) {
      tooltip.style.display = 'none';
      return;
    }

    const tx = curX.invert(mx);
    const ty = yScale.invert(my);

    // Viewport-filtered hit test (iterate only visible pixels)
    const vxMin = curX.invert(M.left);
    const vxMax = curX.invert(W - M.right);
    const vyMin = yScale.invert(H - M.bottom);
    const vyMax = yScale.invert(M.top);

    let found = null, foundFC = null, foundCl = null;

    outer:
    for (const fc of DATA) {
      if (state.hiddenFCs.has(fc.id)) continue;
      for (const cl of fc.clusters) {
        for (const p of cl.pixels) {
          if (p[0] + p[2] <= vxMin || p[0] >= vxMax) continue;
          if (p[1] + p[3] <= vyMin || p[1] >= vyMax) continue;
          if (tx >= p[0] && tx <= p[0] + p[2] &&
              ty >= p[1] && ty <= p[1] + p[3]) {
            found = p; foundFC = fc; foundCl = cl;
            break outer;
          }
        }
      }
    }

    if (found) {
      tooltip.style.display = 'block';
      // Keep tooltip inside viewport
      const tipW = 210, tipH = 130;
      const left = mx + 16 + tipW > W ? mx - tipW - 8 : mx + 16;
      const top  = my - 20 < 0 ? my + 10 : my - 20;
      tooltip.style.left = left + 'px';
      tooltip.style.top  = top  + 'px';
      tooltip.innerHTML =
        `<b>${foundFC.label}</b><br>` +
        `Cluster: #${foundCl.local_id}<br>` +
        `Time: ${found[0].toFixed(5)} s<br>` +
        `Freq: ${(found[1] + found[3] / 2).toFixed(2)} Hz<br>` +
        `Δt: ${found[2].toFixed(5)} s &nbsp; Δf: ${found[3].toFixed(2)} Hz<br>` +
        `Likelihood: ${found[5].toFixed(4)}<br>` +
        `Core: ${found[4] ? '&#10003;' : '&#8212;'}`;
    } else {
      tooltip.style.display = 'none';
    }
  }, 40);
}

// ============================================================
//  Initialise
// ============================================================
function init() {
  assignColors();
  buildSidebar();

  document.getElementById('plot-title').textContent = TITLE;

  wrapper = document.getElementById('chart-wrapper');
  canvas  = document.getElementById('pixel-canvas');
  const svgEl = document.getElementById('axis-svg');

  W = wrapper.clientWidth;
  H = wrapper.clientHeight;
  canvas.width  = W;
  canvas.height = H;
  ctx = canvas.getContext('2d');

  svg = d3.select(svgEl).attr('width', W).attr('height', H);

  // ---- Scales ----
  const bounds = computeBounds();
  const xPad = (bounds.xMax - bounds.xMin) * 0.03 || 0.5;
  const yPad = (bounds.yMax - bounds.yMin) * 0.03 || 5;

  xScale = d3.scaleLinear()
    .domain([bounds.xMin - xPad, bounds.xMax + xPad])
    .range([M.left, W - M.right]);

  yScale = d3.scaleLinear()
    .domain([Math.max(0, bounds.yMin - yPad), bounds.yMax + yPad])
    .range([H - M.bottom, M.top]);

  // ---- Grid + Axes ----
  const xGridG = svg.append('g').attr('class', 'x-grid')
    .attr('transform', `translate(0,${H - M.bottom})`);
  const yGridG = svg.append('g').attr('class', 'y-grid')
    .attr('transform', `translate(${M.left},0)`);

  xAxisG = svg.append('g').attr('class', 'axis x-axis')
    .attr('transform', `translate(0,${H - M.bottom})`);
  yAxisG = svg.append('g').attr('class', 'axis y-axis')
    .attr('transform', `translate(${M.left},0)`);

  // Axis labels
  svg.append('text').attr('class', 'axis-label')
    .attr('x', (M.left + W - M.right) / 2)
    .attr('y', H - 6)
    .attr('text-anchor', 'middle')
    .text('Time (s)');

  svg.append('text').attr('class', 'axis-label')
    .attr('transform', 'rotate(-90)')
    .attr('x', -(M.top + H - M.bottom) / 2)
    .attr('y', 16)
    .attr('text-anchor', 'middle')
    .text('Frequency (Hz)');

  // Pixel count (total)
  document.getElementById('info-counts').textContent =
    `${bounds.total.toLocaleString()} total pixels`;

  // ---- Zoom ----
  zoomBehavior = d3.zoom()
    .scaleExtent([0.02, 5000])
    .on('zoom', event => {
      state.transform = event.transform;
      scheduleRender();
    });

  // Transparent interaction rect (covers chart area, above canvas)
  svg.append('rect')
    .attr('id', 'zoom-rect')
    .attr('x', M.left).attr('y', M.top)
    .attr('width',  W - M.left - M.right)
    .attr('height', H - M.top  - M.bottom)
    .attr('fill', 'transparent')
    .call(zoomBehavior)
    .on('mousemove', onMouseMove)
    .on('mouseleave', () => {
      if (_ttTimer) clearTimeout(_ttTimer);
      document.getElementById('tooltip').style.display = 'none';
    });

  // Reset button
  document.getElementById('btn-reset').addEventListener('click', () => {
    d3.select('#zoom-rect').call(zoomBehavior.transform, d3.zoomIdentity);
  });

  // ---- Resize ----
  new ResizeObserver(() => {
    W = wrapper.clientWidth;
    H = wrapper.clientHeight;
    canvas.width  = W;
    canvas.height = H;
    svg.attr('width', W).attr('height', H);
    xScale.range([M.left, W - M.right]);
    yScale.range([H - M.bottom, M.top]);
    xAxisG.attr('transform', `translate(0,${H - M.bottom})`);
    d3.select('#zoom-rect')
      .attr('width',  W - M.left - M.right)
      .attr('height', H - M.top  - M.bottom);
    scheduleRender();
  }).observe(wrapper);

  scheduleRender();
}

document.addEventListener('DOMContentLoaded', init);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_fragment_clusters(
    fragment_clusters,
    labels: Optional[list] = None,
    output_path: str = "fragment_clusters.html",
    title: str = "Fragment Cluster Visualization",
) -> str:
    """
    Generate a self-contained interactive HTML visualisation of
    :class:`~pycwb.types.network_cluster.FragmentCluster` objects.

    Each ``FragmentCluster`` is shown as a filterable layer.  Within each
    layer every ``Cluster`` is drawn with a distinct colour.  The pixels are
    rendered as time-frequency rectangles whose size reflects the WDM
    resolution (larger ``dt``/``df`` at lower rates).

    Parameters
    ----------
    fragment_clusters : list[FragmentCluster]
        One or more ``FragmentCluster`` objects to visualise.
    labels : list[str], optional
        Human-readable label for each ``FragmentCluster``.  Defaults to
        ``"FragmentCluster 0"``, ``"FragmentCluster 1"``, …
    output_path : str
        Destination path for the generated HTML file.  Parent directories
        are created automatically.
    title : str
        Plot title shown in the browser tab and the info bar.

    Returns
    -------
    str
        Absolute path to the written HTML file.

    Notes
    -----
    * The visualisation requires an internet connection to fetch D3 v7 from
      jsDelivr CDN (``https://cdn.jsdelivr.net``).
    * For very large datasets (> 100 k pixels) consider pre-filtering
      ``fragment_clusters`` to a subset of interest before calling this
      function.
    """
    data = _serialize(fragment_clusters, labels)
    data_json  = json.dumps(data, separators=(",", ":"))
    title_json = json.dumps(title)       # safely quoted JS string literal

    html = (
        _HTML
        .replace("<<<TITLE_RAW>>>", title)
        .replace("<<<TITLE_JSON>>>", title_json)
        .replace("<<<DATA>>>", data_json)
    )

    out = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)

    print(f"Fragment cluster visualisation saved → {out}")
    return out
