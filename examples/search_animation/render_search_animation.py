#!/usr/bin/env python
"""Render a didactic pycWB coherent-search animation.

The search itself lives in ``search_model.py``; this script turns its
products into nine scenes that follow the pycWB native pipeline:

1. projection      h+ -> F+, Fx, arrival delay per detector
2. whitening       coloured noise -> per-layer WDM noise RMS -> whitened data
3. WDM             the same data at several time-frequency resolutions
4. selection       delay-maximised network energy and the bpp threshold
5. clustering      8-neighbour clusters per resolution
6. supercluster    fragments linked across resolutions, subnet cut
7. sky loop        HEALPix scan of the coherent statistic Lo * cc
8. reconstruction  whitened waveform and null stream at the best sky point
9. background      time slides, (rho, netcc) cuts and a FAR bound
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time as timer
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))

import search_model as sm  # noqa: E402

DISPLAY_FMAX = 900.0

BG = "#05070b"
PANEL = "#070a0f"
SPINE = "#273444"
GRID = "#1c2632"
INK = "#f6f8fb"
INK_2 = "#c8d3df"
INK_3 = "#8795a6"
ACCENT = "#FFD43B"
ACCENT_2 = "#FFE873"
IFO_COLORS = {"H1": "#e66767", "L1": "#3987e5", "V1": "#199e70", "K1": "#9085e9"}

# PyCBC/gwpy restyle matplotlib on import; pin the parts the animation relies on.
RC_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["FreeSans", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10.0,
    "axes.grid": False,
    "axes.axisbelow": True,
    "axes.formatter.use_mathtext": True,
    "legend.handlelength": 2.0,
    "legend.numpoints": 1,
    "legend.fancybox": True,
}

SCENES = (
    ("projection", "Signal model: detector projection", 5.0),
    ("whitening", "Data conditioning: whitening", 7.0),
    ("wdm", "Multi-resolution WDM transform", 6.0),
    ("selection", "Coherent pixel selection", 7.0),
    ("clustering", "Clustering per resolution", 5.0),
    ("supercluster", "Supercluster across resolutions", 6.0),
    ("sky", "Sky loop: coherent likelihood", 12.0),
    ("reconstruction", "Waveform reconstruction", 7.0),
    ("background", "Significance from time slides", 8.0),
)
BREADCRUMB = ("project", "whiten", "WDM", "select", "cluster", "supercluster", "sky loop", "reconstruct", "background")


def parse_args() -> argparse.Namespace:
    defaults = sm.SearchConfig()
    parser = argparse.ArgumentParser(
        description="Render a pycWB coherent-search animation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default="tests/logo/cWB_logo_waveform.txt", type=Path)
    parser.add_argument("--out", default="examples/search_animation/output", type=Path)
    parser.add_argument("--ifos", nargs="+", default=list(defaults.ifos))
    parser.add_argument("--levels", nargs="+", type=int, default=list(defaults.levels), help="WDM layers M per resolution")
    parser.add_argument("--noise-sigma", default=defaults.noise_sigma, type=float)
    parser.add_argument("--noise-seed", default=defaults.noise_seed, type=int)
    parser.add_argument("--bpp", default=defaults.bpp, type=float, help="black-pixel probability of the selection threshold")
    parser.add_argument("--nside", default=None, type=int, help="HEALPix nside (default 16 for two detectors, 64 otherwise)")
    parser.add_argument("--lags", default=defaults.n_lags, type=int, help="number of circular time slides")
    parser.add_argument("--lag-step", default=defaults.lag_step, type=float)
    parser.add_argument("--width", default=1280, type=int)
    parser.add_argument("--height", default=720, type=int)
    parser.add_argument("--fps", default=24, type=int)
    parser.add_argument("--duration", default=60.0, type=float)
    parser.add_argument("--scenes", nargs="+", choices=[s[0] for s in SCENES], default=[s[0] for s in SCENES])
    parser.add_argument("--frames-limit", default=None, type=int)
    parser.add_argument("--format", nargs="*", choices=("mp4", "gif"), default=["mp4", "gif"])
    parser.add_argument("--gif-fps", default=10, type=int)
    parser.add_argument("--gif-width", default=854, type=int)
    parser.add_argument("--stills", nargs="*", type=float, default=None, help="also save PNG stills of every scene at these progress values")
    parser.add_argument("--dpi", default=100, type=int)
    args = parser.parse_args()
    if args.levels and 64 not in args.levels:
        parser.error("--levels must include 64 (the display resolution)")
    return args


# ---------------------------------------------------------------------------
# colour maps


def make_tf_cmap() -> LinearSegmentedColormap:
    cmap = LinearSegmentedColormap.from_list(
        "python_wdm",
        ["#05070b", "#10263f", "#3776AB", "#4B8BBE", "#6FA8DC", "#FFE873", "#FFD43B", "#F4A300"],
    )
    cmap.set_bad((0.02, 0.025, 0.035, 1.0))
    return cmap


def make_sky_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "sky_statistic", ["#1a2b40", "#1f3b5e", "#3776AB", "#6FA8DC", "#FFE873", "#FFD43B", "#F4A300"]
    )


def make_diverging_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("coherent", ["#3987e5", "#1c3f6e", "#383835", "#8a6414", "#F4A300"])


def make_coverage_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("coverage", ["#10263f", "#3776AB", "#6FA8DC", "#FFE873"])


# ---------------------------------------------------------------------------
# render context: everything derived once from the search result


def _smooth(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def ramp(p: float, start: float, length: float) -> float:
    return _smooth((p - start) / length)


@dataclass
class TFCrop:
    image: np.ndarray
    extent: list[float]


def crop_tf(arr: np.ndarray, dt: float, df: float, window: tuple[float, float]) -> TFCrop:
    n_time = arr.shape[-1]
    j0 = max(int(np.floor(window[0] / dt)) - 1, 0)
    j1 = min(int(np.ceil(window[1] / dt)) + 1, n_time - 1)
    sub = arr[..., j0 : j1 + 1]
    n_freq = arr.shape[-2]
    extent = [(j0 - 0.5) * dt - window[0], (j1 + 0.5) * dt - window[0], -0.5 * df, (n_freq - 0.5) * df]
    return TFCrop(sub, extent)


def log_norm(energy: np.ndarray, ref: float) -> np.ndarray:
    return np.clip(np.log1p(np.maximum(energy, 0.0)) / np.log1p(ref), 0.0, 1.0)


def mollweide_forward(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    theta = lat.copy()
    for _ in range(12):
        f = 2 * theta + np.sin(2 * theta) - np.pi * np.sin(lat)
        fp = 2 + 2 * np.cos(2 * theta)
        theta = theta - np.where(np.abs(fp) > 1e-9, f / np.maximum(fp, 1e-9), 0.0)
    x = 2 * np.sqrt(2) / np.pi * lon * np.cos(theta)
    y = np.sqrt(2) * np.sin(theta)
    return x, y


def ra_to_lon(ra: np.ndarray | float) -> np.ndarray:
    """Sky longitude with RA increasing to the left, RA = 12 h at the centre."""
    return np.pi - np.mod(np.asarray(ra, dtype=float), 2 * np.pi)


class RenderContext:
    def __init__(self, result: sm.SearchResult, args: argparse.Namespace):
        import healpy as hp

        self.r = result
        cfg = result.config
        self.cfg = cfg
        self.ifos = list(cfg.ifos)
        self.n_det = len(self.ifos)
        self.fs = result.sample_rate
        self.window = result.window
        w0, w1 = self.window
        i0 = int(round(w0 * self.fs))
        i1 = int(round(w1 * self.fs))
        self.i0, self.i1 = i0, i1
        self.t = result.time[i0:i1] - w0
        self.t_end = float(w1 - w0)
        self.tf_cmap = make_tf_cmap()
        self.sky_cmap = make_sky_cmap()
        self.div_cmap = make_diverging_cmap()
        self.cov_cmap = make_coverage_cmap()
        self.colors = [IFO_COLORS.get(ifo, "#c8d3df") for ifo in self.ifos]

        zero = result.zero_lag
        self.zero = zero
        self.levels = list(cfg.levels)
        self.display = self.levels.index(cfg.display_level)
        disp = zero.resolutions[self.display]

        # scene 2: raw and whitened display maps
        white_crop = crop_tf(result.white_display, disp.dt, disp.df, self.window)
        raw_crop = crop_tf(result.raw_display, disp.dt, disp.df, self.window)
        self.energy_ref = float(np.percentile(white_crop.image, 99.7))
        self.white_tf = [TFCrop(log_norm(white_crop.image[d], self.energy_ref), white_crop.extent) for d in range(self.n_det)]
        raw_ref = float(np.percentile(raw_crop.image, 99.9))
        self.raw_tf = [TFCrop(log_norm(raw_crop.image[d], raw_ref), raw_crop.extent) for d in range(self.n_det)]

        # scene 3-5: per-resolution maps
        self.res_tf = []
        self.res_max = []
        self.res_net = []
        self.res_selected = []
        self.res_core = []
        self.res_labels = []
        for res in zero.resolutions:
            energy = sm.pixel_energy(res.coeff)
            self.res_tf.append([TFCrop(log_norm(c.image, self.energy_ref), c.extent) for c in [crop_tf(energy[d], res.dt, res.df, self.window) for d in range(self.n_det)]])
            self.res_max.append([crop_tf(res.energy_max[d], res.dt, res.df, self.window) for d in range(self.n_det)])
            self.res_net.append(crop_tf(res.net_energy, res.dt, res.df, self.window))
            self.res_selected.append(crop_tf(res.selected, res.dt, res.df, self.window))
            self.res_core.append(crop_tf(res.core, res.dt, res.df, self.window))
            self.res_labels.append(crop_tf(res.labels, res.dt, res.df, self.window))
        self.net_ref = float(np.percentile(self.res_net[self.display].image, 99.7))

        total = result.time[-1] + 1.0 / self.fs
        noise_mask = sm.band_mask(disp.n_freq, disp.df, cfg)[:, None] & sm.noise_time_mask(disp.n_time, disp.dt, cfg, total, self.window)[None, :]
        self.hist_noise = disp.net_energy[noise_mask]
        band = sm.band_mask(disp.n_freq, disp.df, cfg)
        in_window = np.zeros(disp.n_time, dtype=bool)
        tt = np.arange(disp.n_time) * disp.dt
        in_window[(tt >= w0) & (tt <= w1)] = True
        self.hist_window = disp.net_energy[band[:, None] & in_window[None, :]]

        # clusters inside the display window, grown from their loudest pixel
        self.cluster_boxes = []
        self.cluster_depth = []
        for r, res in enumerate(zero.resolutions):
            labels = self.res_labels[r].image
            net = self.res_net[r].image
            depth = np.full(labels.shape, np.inf)
            boxes = []
            ext = self.res_labels[r].extent
            for label, sl in enumerate(ndimage.find_objects(labels), start=1):
                if sl is None:
                    continue
                mask = labels == label
                depth[mask] = bfs_depth(mask, np.where(mask, net, -np.inf))
                f_idx, t_idx = np.nonzero(mask)
                boxes.append((
                    ext[0] + t_idx.min() * res.dt, ext[0] + (t_idx.max() + 1) * res.dt,
                    (f_idx.min() - 0.5) * res.df, (f_idx.max() + 0.5) * res.df, int(mask.sum()), label,
                ))
            boxes.sort(key=lambda b: b[0])
            finite = np.isfinite(depth)
            if np.any(finite):
                for label in np.unique(labels[labels > 0]):
                    m = labels == label
                    depth[m] = depth[m] / max(float(np.max(depth[m])), 1.0)
            self.cluster_boxes.append(boxes)
            self.cluster_depth.append(depth)

        # scene 6: supercluster geometry
        self.event = result.event
        sc = zero.superclusters[self.event.cluster]
        self.sc = sc
        self.fine_t = np.linspace(0.0, self.t_end, 1025)
        self.fine_f = np.linspace(0.0, DISPLAY_FMAX, 451)
        tc = 0.5 * (self.fine_t[:-1] + self.fine_t[1:]) + w0
        fc = 0.5 * (self.fine_f[:-1] + self.fine_f[1:])
        self.coverage = []
        self.sc_masks = []
        for r, res in enumerate(zero.resolutions):
            mask = np.zeros((res.n_freq, res.n_time), dtype=bool)
            sel = sc.res == r
            mask[sc.f_idx[sel], sc.t_idx[sel]] = True
            self.sc_masks.append(crop_tf(mask, res.dt, res.df, self.window))
            jt = np.clip(np.rint(tc / res.dt).astype(int), 0, res.n_time - 1)
            jf = np.clip(np.rint(fc / res.df).astype(int), 0, res.n_freq - 1)
            self.coverage.append(mask[np.ix_(jf, jt)])
        self.frag_centroids = {}
        for i in sc.fragments:
            fr = zero.fragments[i]
            res = zero.resolutions[fr.res]
            self.frag_centroids[i] = (float(np.mean(fr.t_idx * res.dt)) - w0, float(np.mean(fr.f_idx * res.df)))
        self.sc_links = [(a, b) for a, b in zero.links if a in self.frag_centroids and b in self.frag_centroids]
        self.rejected = []
        for other in zero.superclusters:
            if other.passed:
                continue
            t_mean = float(np.mean(other.time)) - w0
            f_mean = float(np.mean(other.freq))
            if 0.0 <= t_mean <= self.t_end and f_mean <= DISPLAY_FMAX:
                reason = f"{other.res.size} px" if other.res.size < cfg.min_pixels else f"subnet {other.subnet:.2f}"
                self.rejected.append((t_mean, f_mean, reason))

        # scene 7: sky
        scan = zero.sky[self.event.cluster]
        self.scan = scan
        self.norm = self.event.norm
        self.n_sky = scan.ra.size
        self.scan_order = np.arange(self.n_sky)
        self.rank = np.empty(self.n_sky, dtype=int)
        self.rank[self.scan_order] = np.arange(self.n_sky)
        stat = np.maximum(scan.statistic, 0.0)
        self.stat_norm = stat / max(float(stat.max()), 1e-12)
        nx, ny = 560, 280
        xs = np.linspace(-2 * np.sqrt(2), 2 * np.sqrt(2), nx)
        ys = np.linspace(-np.sqrt(2), np.sqrt(2), ny)
        X, Y = np.meshgrid(xs, ys)
        inside = X ** 2 / 8 + Y ** 2 / 2 <= 1.0
        theta = np.arcsin(np.clip(Y / np.sqrt(2), -1, 1))
        lat = np.arcsin(np.clip((2 * theta + np.sin(2 * theta)) / np.pi, -1, 1))
        lon = np.pi * X / (2 * np.sqrt(2) * np.maximum(np.cos(theta), 1e-9))
        inside &= np.abs(lon) <= np.pi
        ra = np.mod(np.pi - lon, 2 * np.pi)
        self.sky_inside = inside
        self.sky_idx = hp.ang2pix(cfg.nside, np.clip(0.5 * np.pi - lat, 0, np.pi), ra)
        self.sky_extent = [xs[0], xs[-1], ys[0], ys[-1]]
        self.sky_xy = mollweide_forward(ra_to_lon(scan.ra), scan.dec)
        self.src_xy = mollweide_forward(ra_to_lon(cfg.source_ra), cfg.source_dec)
        self.best_xy = (self.sky_xy[0][scan.best], self.sky_xy[1][scan.best])
        self.delay_imgs = [np.where(inside, scan.rel_delay[d][self.sky_idx], np.nan) for d in range(1, self.n_det)]
        coherent_best, _ = sm.sky_pixel_map(zero, self.event, scan.best, cfg)
        coherent_best = coherent_best / self.norm
        self.pixel_ec_ref = float(np.percentile(np.abs(coherent_best[coherent_best != 0]), 90)) if np.any(coherent_best) else 1.0
        self.sc_display_mask = self.sc_masks[self.display]
        self.coherent_cache: dict[int, TFCrop] = {}

        # scene 8
        self.recon = result.reconstructed[:, i0:i1]
        self.inj = result.white_clean[:, i0:i1]
        self.white = result.whitened[:, i0:i1]
        self.overlap = [
            float(np.dot(self.recon[d], self.inj[d]) / max(np.linalg.norm(self.recon[d]) * np.linalg.norm(self.inj[d]), 1e-12))
            for d in range(self.n_det)
        ]
        self.recon_tf = crop_tf(result.recon_energy, disp.dt, disp.df, self.window)
        self.null_tf = crop_tf(result.null_energy, disp.dt, disp.df, self.window)

        # scene 9
        self.lag_triggers = [[(t.rho, t.net_cc, t.passed) for t in lag] for lag in result.background]
        all_bg = [t for lag in result.background for t in lag]
        self.n_background = len(all_bg)
        live = (total - 2 * cfg.edge_seconds)
        self.livetime = live * len(result.lags)
        self.n_louder = sum(1 for t in all_bg if t.passed and t.rho >= self.event.rho)
        self.n_passed_bg = sum(1 for t in all_bg if t.passed)
        dec = 4
        self.full_t = result.time[::dec]
        self.full_white = result.whitened[:, ::dec]
        self.full_signal = result.white_clean[:, ::dec]

    def coherent_map(self, sky_index: int) -> TFCrop:
        if sky_index not in self.coherent_cache:
            res = self.zero.resolutions[self.display]
            coherent, _ = sm.sky_pixel_map(self.zero, self.event, sky_index, self.cfg)
            self.coherent_cache[sky_index] = crop_tf(coherent / self.norm, res.dt, res.df, self.window)
        return self.coherent_cache[sky_index]


# ---------------------------------------------------------------------------
# drawing helpers


def style_axes(ax: plt.Axes, title: str | None = None, size: float = 10) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(SPINE)
        spine.set_linewidth(0.8)
    ax.tick_params(colors="#a8b3c2", labelsize=8, length=3)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    if title:
        ax.set_title(title, color=INK, fontsize=size, loc="left", pad=4)


def show_tf(ax: plt.Axes, crop: TFCrop, cmap, vmin: float = 0.0, vmax: float = 1.0, alpha: float = 1.0, image=None) -> None:
    ax.imshow(
        crop.image if image is None else image,
        extent=crop.extent, origin="lower", aspect="auto", interpolation="nearest",
        cmap=cmap, vmin=vmin, vmax=vmax, alpha=float(np.clip(alpha, 0.0, 1.0)),
    )


def tf_axes(ax: plt.Axes, ctx: RenderContext, xlabel: bool = False, ylabel: bool = True, fmax: float = DISPLAY_FMAX) -> None:
    ax.set_xlim(0.0, ctx.t_end)
    ax.set_ylim(0.0, fmax)
    if ylabel:
        ax.set_ylabel("Hz", fontsize=8)
    else:
        ax.set_yticklabels([])
    if xlabel:
        ax.set_xlabel("time from GPS reference (s)", fontsize=8)
    else:
        ax.set_xticklabels([])


def draw_header(fig: plt.Figure, scene_index: int, title: str) -> None:
    fig.text(0.045, 0.965, f"{scene_index + 1}  {title}", color=INK, fontsize=16, fontweight="bold", va="top")
    n = len(BREADCRUMB)
    x0, x1, y = 0.045, 0.955, 0.892
    xs = np.linspace(x0, x1, n)
    fig.add_artist(plt.Line2D([x0, x1], [y, y], color=SPINE, lw=1.0, transform=fig.transFigure, zorder=1))
    fig.add_artist(plt.Line2D([x0, xs[scene_index]], [y, y], color="#3776AB", lw=1.6, transform=fig.transFigure, zorder=1))
    for i, (x, label) in enumerate(zip(xs, BREADCRUMB)):
        if i < scene_index:
            face, edge, ink = "#17304d", "#3776AB", INK_2
        elif i == scene_index:
            face, edge, ink = ACCENT, ACCENT, "#05070b"
        else:
            face, edge, ink = "#0b1118", SPINE, INK_3
        ha = "left" if i == 0 else ("right" if i == n - 1 else "center")
        fig.text(
            x, y, f" {i + 1} {label} ", color=ink, fontsize=8, ha=ha, va="center", zorder=3,
            fontweight="bold" if i == scene_index else "normal",
            bbox=dict(boxstyle="round,pad=0.28", facecolor=face, edgecolor=edge, linewidth=0.9),
        )


def draw_fade(fig: plt.Figure, alpha: float) -> None:
    if alpha <= 0.002:
        return
    fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure, facecolor=BG, edgecolor="none", alpha=float(min(alpha, 1.0)), zorder=100))


def card(ax: plt.Axes, alpha: float = 1.0, edge: str = SPINE) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.0,rounding_size=0.03", transform=ax.transAxes,
            facecolor="#0b1118", edgecolor=edge, linewidth=1.0, alpha=0.95 * alpha, clip_on=False,
        )
    )


def text_lines(ax: plt.Axes, lines: list[tuple[str, str]], x: float, y: float, dy: float, alpha: float, size: float = 10) -> None:
    """Draw (label, value) rows; the label in secondary ink, the value in primary ink."""
    for i, (label, value) in enumerate(lines):
        yy = y - i * dy
        ax.text(x, yy, label, color=INK_3, fontsize=size - 1, va="center", ha="left", alpha=alpha, transform=ax.transAxes)
        ax.text(x + 0.42, yy, value, color=INK, fontsize=size, va="center", ha="left", alpha=alpha, transform=ax.transAxes)


# ---------------------------------------------------------------------------
# scene 1: projection


def scene_projection(fig: plt.Figure, ctx: RenderContext, p: float) -> None:
    r = ctx.r
    draw_h = ramp(p, 0.0, 0.35)
    draw_d = ramp(p, 0.28, 0.35)
    show_card = ramp(p, 0.45, 0.25)
    show_zoom = ramp(p, 0.6, 0.3)
    left, width = 0.06, 0.55
    rows = 1 + ctx.n_det
    top, bottom, gap = 0.84, 0.09, 0.05
    height = (top - bottom - gap * (rows - 1)) / rows
    t = ctx.t
    hp = r.hplus / np.max(np.abs(r.hplus))
    clean = r.clean[:, ctx.i0 : ctx.i1]

    ax = fig.add_axes([left, top - height, width, height])
    style_axes(ax, r"source waveform $h_+(t)$  ($h_\times = 0$)")
    n = int(draw_h * t.size)
    ax.plot(t[:n], hp[:n], color=INK, lw=0.7)
    ax.set_xlim(0, ctx.t_end)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xticklabels([])
    ax.grid(color=GRID, lw=0.4)

    n = int(draw_d * t.size)
    for d in range(ctx.n_det):
        y0 = top - (d + 2) * height - (d + 1) * gap
        ax = fig.add_axes([left, y0, width, height])
        fp, fc = r.antenna[d]
        style_axes(ax, rf"{ctx.ifos[d]} response  $F_+ h_+(t-\tau)$,  $F_+={fp:+.2f}$")
        ax.plot(t[:n], clean[d, :n], color=ctx.colors[d], lw=0.7)
        ax.set_xlim(0, ctx.t_end)
        ax.set_ylim(-1.15, 1.15)
        ax.grid(color=GRID, lw=0.4)
        if d == ctx.n_det - 1:
            ax.set_xlabel("time from GPS reference (s)", fontsize=8)
        else:
            ax.set_xticklabels([])

    # card with antenna patterns and delays
    ax = fig.add_axes([0.66, 0.53, 0.30, 0.31])
    card(ax, show_card)
    ax.text(0.05, 0.9, r"$h_d(t)=F_{+,d}\,h_+(t-\tau_d)+F_{\times,d}\,h_\times(t-\tau_d)$", color=INK, fontsize=10.5, transform=ax.transAxes, alpha=show_card, va="center")
    ax.text(
        0.05, 0.76, rf"sky $(\alpha,\delta)=({ctx.cfg.source_ra:.2f}, {ctx.cfg.source_dec:.2f})$ rad,  $\psi={ctx.cfg.source_polarization:.1f}$",
        color=INK_2, fontsize=9, transform=ax.transAxes, alpha=show_card, va="center",
    )
    header_y = 0.6
    for x, label in zip((0.05, 0.28, 0.52, 0.76), ("detector", r"$F_+$", r"$F_\times$", r"$\tau$ (ms)")):
        ax.text(x, header_y, label, color=INK_3, fontsize=9, transform=ax.transAxes, alpha=show_card, va="center")
    for d in range(ctx.n_det):
        yy = header_y - 0.14 * (d + 1)
        fp, fc = r.antenna[d]
        ax.text(0.05, yy, ctx.ifos[d], color=ctx.colors[d], fontsize=10, fontweight="bold", transform=ax.transAxes, alpha=show_card, va="center")
        for x, val in zip((0.28, 0.52, 0.76), (f"{fp:+.3f}", f"{fc:+.3f}", f"{r.arrival_delay[d] * 1e3:+.2f}")):
            ax.text(x, yy, val, color=INK, fontsize=10, transform=ax.transAxes, alpha=show_card, va="center")
    if ctx.n_det > 1:
        pairs = ",  ".join(
            rf"$\Delta\tau_{{{ctx.ifos[d]}-{ctx.ifos[0]}}}={(r.arrival_delay[d] - r.arrival_delay[0]) * 1e3:+.2f}$ ms" for d in range(1, ctx.n_det)
        )
        ax.text(0.05, 0.08, pairs, color=ACCENT_2, fontsize=10 if ctx.n_det == 2 else 9, transform=ax.transAxes, alpha=show_card, va="center")

    # zoom showing the delay
    if show_zoom <= 0.01:
        return
    ax = fig.add_axes([0.66, 0.09, 0.30, 0.34])
    style_axes(ax, "zoom: same wavefront, delayed and rescaled", size=9)
    for spine in ax.spines.values():
        spine.set_alpha(show_zoom)
    if show_zoom > 0.01:
        peak = int(np.argmax(np.abs(clean[0])))
        half = int(0.018 * ctx.fs)
        sl = slice(max(peak - half, 0), min(peak + half, t.size))
        for d in range(ctx.n_det):
            ax.plot(t[sl] * 1e3, clean[d, sl], color=ctx.colors[d], lw=1.4, alpha=show_zoom, label=ctx.ifos[d])
        if ctx.n_det > 1:
            t_h = t[peak] * 1e3
            t_l = t_h + (r.arrival_delay[1] - r.arrival_delay[0]) * 1e3
            ymax = 1.05
            for tt, c in ((t_h, ctx.colors[0]), (t_l, ctx.colors[1])):
                ax.axvline(tt, color=c, lw=0.8, ls=":", alpha=show_zoom)
            ax.annotate(
                "", xy=(t_l, 0.92 * ymax), xytext=(t_h, 0.92 * ymax),
                arrowprops=dict(arrowstyle="<->", color=ACCENT_2, lw=1.1, alpha=show_zoom),
            )
            ax.text(0.5 * (t_h + t_l), 0.98 * ymax, f"{abs(t_l - t_h):.1f} ms", color=ACCENT_2, fontsize=9, ha="center", va="bottom", alpha=show_zoom)
        ax.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=INK_2)
        ax.set_xlim(t[sl][0] * 1e3, t[sl][-1] * 1e3)
    ax.set_ylim(-1.15, 1.25)
    ax.set_xlabel("time (ms)", fontsize=8)
    ax.grid(color=GRID, lw=0.4)


# ---------------------------------------------------------------------------
# scene 2: whitening


def robust_scale(x: np.ndarray) -> np.ndarray:
    return x / max(float(np.percentile(np.abs(x), 99.7)), 1e-12)


def scene_whitening(fig: plt.Figure, ctx: RenderContext, p: float) -> None:
    r = ctx.r
    show_rms = ramp(p, 0.15, 0.35)
    mix = ramp(p, 0.52, 0.33)
    raw = r.raw[:, ctx.i0 : ctx.i1]
    white = r.whitened[:, ctx.i0 : ctx.i1]
    gs = fig.add_gridspec(
        2 * ctx.n_det, 1, left=0.06, right=0.64, top=0.84, bottom=0.08, hspace=0.25 + 0.15 * (ctx.n_det - 2),
        height_ratios=[1.0, 1.5] * ctx.n_det,
    )
    for d in range(ctx.n_det):
        ax = fig.add_subplot(gs[2 * d, 0])
        label = "raw strain (coloured noise)" if mix < 0.5 else "whitened strain"
        style_axes(ax, f"{ctx.ifos[d]} {label}")
        if mix < 0.999:
            ax.plot(ctx.t, robust_scale(raw[d]), color="#a8b3c2", lw=0.5, alpha=1.0 - mix)
        if mix > 0.001:
            ax.plot(ctx.t, robust_scale(white[d]), color=ctx.colors[d], lw=0.5, alpha=mix)
        ax.set_xlim(0, ctx.t_end)
        ax.set_ylim(-1.2, 1.2)
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.grid(color=GRID, lw=0.4)

        ax = fig.add_subplot(gs[2 * d + 1, 0])
        style_axes(ax, f"{ctx.ifos[d]} WDM energy (M = {ctx.cfg.display_level})" + ("  after whitening" if mix >= 0.5 else ""))
        show_tf(ax, ctx.raw_tf[d], ctx.tf_cmap, alpha=1.0 - mix)
        if mix > 0.001:
            show_tf(ax, ctx.white_tf[d], ctx.tf_cmap, alpha=mix)
        tf_axes(ax, ctx, xlabel=(d == ctx.n_det - 1))

    ax = fig.add_axes([0.71, 0.24, 0.25, 0.58])
    style_axes(ax, "per-layer noise RMS", size=10)
    freq = np.maximum(r.white_freq, 1.0)
    keep = r.white_freq > 0
    ax.axvspan(ctx.cfg.f_low, ctx.cfg.f_high, color="#3776AB", alpha=0.10)
    ax.text(ctx.cfg.f_low * 1.1, 0.4, "analysis band", color=INK_3, fontsize=8, va="bottom")
    ax.plot(freq[keep], r.asd_at_layers[keep], color=INK_3, lw=1.0, ls="--", label="injected ASD shape")
    n = int(show_rms * keep.sum())
    idx = np.nonzero(keep)[0][:n]
    for d in range(ctx.n_det):
        ax.plot(freq[idx], r.white_rms[d][idx], color=ctx.colors[d], lw=1.4, label=f"{ctx.ifos[d]} estimate")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(6, 2048)
    ax.set_ylim(0.3, float(np.max(r.white_rms)) * 4)
    ax.set_xlabel("frequency (Hz)", fontsize=8)
    ax.set_ylabel(r"$\sigma_f$ (relative)", fontsize=8)
    ax.grid(color=GRID, lw=0.4, which="both")
    ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK_2)

    ax = fig.add_axes([0.71, 0.08, 0.25, 0.1])
    card(ax, 1.0)
    ax.text(0.05, 0.68, r"$\sigma_f=\sqrt{0.7191\;\mathrm{median}_t\,(w_{00}^2+w_{90}^2)}$", color=INK, fontsize=10, transform=ax.transAxes, va="center")
    ax.text(0.05, 0.28, r"$\tilde w(t,f)=w(t,f)/\sigma_f$" + f"   (M = {ctx.cfg.whiten_level})", color=INK_2, fontsize=9, transform=ax.transAxes, va="center", alpha=ramp(p, 0.45, 0.2))


# ---------------------------------------------------------------------------
# scene 3: multi-resolution WDM


def scene_wdm(fig: plt.Figure, ctx: RenderContext, p: float) -> None:
    n_res = len(ctx.levels)
    left, right, gap = 0.06, 0.965, 0.018
    width = (right - left - gap * (n_res - 1)) / n_res
    zoom_t = (0.60, 0.86)
    zoom_f = (120.0, 460.0)
    for r, level in enumerate(ctx.levels):
        alpha = ramp(p, 0.05 + 0.17 * r, 0.2)
        res = ctx.zero.resolutions[r]
        x = left + r * (width + gap)
        ax = fig.add_axes([x, 0.50, width, 0.32])
        style_axes(ax, f"{ctx.ifos[0]} · M = {level}:  Δt = {res.dt * 1e3:.1f} ms,  Δf = {res.df:.0f} Hz", size=9)
        if alpha > 0.01:
            show_tf(ax, ctx.res_tf[r][0], ctx.tf_cmap, alpha=alpha)
            ax.add_patch(Rectangle((zoom_t[0], zoom_f[0]), zoom_t[1] - zoom_t[0], zoom_f[1] - zoom_f[0], fill=False, edgecolor=INK, lw=0.8, alpha=alpha))
        tf_axes(ax, ctx, xlabel=True, ylabel=(r == 0))

        ax = fig.add_axes([x, 0.08, width, 0.30])
        style_axes(ax, "zoom on the white box: pixel tiling", size=9)
        if alpha > 0.01:
            show_tf(ax, ctx.res_tf[r][0], ctx.tf_cmap, alpha=alpha)
            n_t = int(round((zoom_t[1] - zoom_t[0]) / res.dt))
            n_f = int(round((zoom_f[1] - zoom_f[0]) / res.df))
            if n_t * n_f < 900:
                t_edges = (np.arange(np.floor((zoom_t[0] + ctx.window[0]) / res.dt - 0.5), np.ceil((zoom_t[1] + ctx.window[0]) / res.dt + 0.5)) + 0.5) * res.dt - ctx.window[0]
                f_edges = (np.arange(np.floor(zoom_f[0] / res.df - 0.5), np.ceil(zoom_f[1] / res.df + 0.5)) + 0.5) * res.df
                for te in t_edges:
                    ax.axvline(te, color=BG, lw=0.4, alpha=0.6 * alpha)
                for fe in f_edges:
                    ax.axhline(fe, color=BG, lw=0.4, alpha=0.6 * alpha)
        ax.set_xlim(*zoom_t)
        ax.set_ylim(*zoom_f)
        ax.set_xlabel("time (s)", fontsize=8)
        if r == 0:
            ax.set_ylabel("Hz", fontsize=8)
        else:
            ax.set_yticklabels([])
    fig.text(
        0.965, 0.855, f"Δt·Δf = 1/2 at every level · every detector gets the same {n_res} maps",
        color=INK_3, fontsize=9, ha="right", va="center", alpha=ramp(p, 0.7, 0.2),
    )


# ---------------------------------------------------------------------------
# scene 4: selection


def scene_selection(fig: plt.Figure, ctx: RenderContext, p: float) -> None:
    show_det = ramp(p, 0.0, 0.2)
    show_net = ramp(p, 0.2, 0.2)
    show_hist = ramp(p, 0.35, 0.25)
    show_sel = ramp(p, 0.62, 0.28)
    r = ctx.display
    res = ctx.zero.resolutions[r]
    delay_ms = sm.max_light_travel_time(ctx.cfg) * 1e3

    n_det = ctx.n_det
    width = (0.90 - 0.02 * (n_det - 1)) / n_det
    for d in range(n_det):
        ax = fig.add_axes([0.06 + d * (width + 0.02), 0.52, width, 0.30])
        style_axes(ax, rf"{ctx.ifos[d]}: $E^{{\max}}_{{{ctx.ifos[d]}}}(t,f)=\max_{{|\tau|\leq {delay_ms:.0f}\,\mathrm{{ms}}}}E(t+\tau,f)$", size=9.5)
        crop = ctx.res_max[r][d]
        show_tf(ax, crop, ctx.tf_cmap, alpha=show_det, image=log_norm(crop.image, ctx.energy_ref))
        tf_axes(ax, ctx, xlabel=False, ylabel=(d == 0))

    ax = fig.add_axes([0.06, 0.08, 0.53, 0.36])
    style_axes(ax, r"network energy $E_{\rm net}=\sum_d E^{\max}_d$" + (f"  ·  {int(ctx.res_selected[r].image.sum())} pixels kept" if show_sel > 0.5 else ""), size=9.5)
    net = log_norm(ctx.res_net[r].image, ctx.net_ref)
    if show_sel > 0:
        net = np.where(ctx.res_selected[r].image, net, net * (1.0 - 0.85 * show_sel))
    show_tf(ax, ctx.res_net[r], ctx.tf_cmap, alpha=show_net, image=net)
    if show_sel > 0.3:
        sel = ctx.res_selected[r]
        centers_t = np.linspace(sel.extent[0], sel.extent[1], sel.image.shape[1] + 1)
        centers_t = 0.5 * (centers_t[:-1] + centers_t[1:])
        centers_f = np.arange(sel.image.shape[0]) * res.df
        ax.contour(centers_t, centers_f, sel.image.astype(float), levels=[0.5], colors=[INK], linewidths=0.6, alpha=show_sel)
    tf_axes(ax, ctx, xlabel=True)

    ax = fig.add_axes([0.66, 0.225, 0.30, 0.215])
    style_axes(ax, "network-energy distribution", size=9.5)
    if show_hist > 0.01:
        bins = np.logspace(np.log10(0.5), np.log10(max(ctx.hist_window.max(), 10) * 1.1), 45)
        counts_n, _ = np.histogram(ctx.hist_noise, bins=bins)
        counts_w, _ = np.histogram(ctx.hist_window, bins=bins)
        scale = ctx.hist_window.size / max(ctx.hist_noise.size, 1)
        ax.stairs(np.maximum(counts_n * scale, 1e-3), bins, fill=True, color="#3a4656", alpha=show_hist, label="noise-only data (scaled)")
        ax.stairs(np.maximum(counts_w, 1e-3), bins, color=ACCENT, lw=1.2, alpha=show_hist, label="analysed window")
        thr = res.threshold
        ax.axvline(thr, color=INK, lw=1.1, alpha=show_hist)
        ax.axvline(2 * thr, color=INK, lw=0.8, ls="--", alpha=show_hist)
        ax.text(thr * 1.06, 0.97, r"$E_o$", color=INK, fontsize=9, va="top", transform=ax.get_xaxis_transform(), alpha=show_hist)
        ax.text(2 * thr * 1.06, 0.97, r"$2E_o$", color=INK, fontsize=9, va="top", transform=ax.get_xaxis_transform(), alpha=show_hist)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.5, None)
        ax.legend(loc="upper right", fontsize=7.5, frameon=False, labelcolor=INK_2, bbox_to_anchor=(1.0, 0.86))
    ax.set_xlabel(r"$E_{\rm net}$ per pixel", fontsize=8, labelpad=1)
    ax.grid(color=GRID, lw=0.4)

    ax = fig.add_axes([0.66, 0.05, 0.30, 0.07])
    ax.axis("off")
    counts = ",  ".join(f"M{lv}: {int(ctx.res_selected[i].image.sum())}" for i, lv in enumerate(ctx.levels))
    ax.text(0.0, 0.85, rf"$P_{{\rm noise}}(E_{{\rm net}}>E_o)=\mathrm{{bpp}}={ctx.cfg.bpp:g}$", color=INK, fontsize=9, transform=ax.transAxes, va="center", alpha=show_hist)
    ax.text(0.0, 0.45, r"kept: $E_{\rm net}\geq 2E_o$, or $\geq E_o$ with a neighbour above $E_o$", color=INK_3, fontsize=8, transform=ax.transAxes, va="center", alpha=show_sel)
    ax.text(0.0, 0.05, "pixels kept in the window  " + counts, color=INK_3, fontsize=8, transform=ax.transAxes, va="center", alpha=show_sel)


# ---------------------------------------------------------------------------
# scene 5: clustering


def bfs_depth(mask: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """Breadth-first 8-neighbour distance from the loudest pixel of ``mask``."""
    from collections import deque

    depth = np.full(mask.shape, np.inf)
    seed = np.unravel_index(int(np.argmax(energy)), mask.shape)
    depth[seed] = 0.0
    queue = deque([seed])
    n_f, n_t = mask.shape
    while queue:
        f, t = queue.popleft()
        for df in (-1, 0, 1):
            for dt in (-1, 0, 1):
                ff, tt = f + df, t + dt
                if 0 <= ff < n_f and 0 <= tt < n_t and mask[ff, tt] and depth[ff, tt] == np.inf:
                    depth[ff, tt] = depth[f, t] + 1
                    queue.append((ff, tt))
    return depth[mask]


def scene_clustering(fig: plt.Figure, ctx: RenderContext, p: float) -> None:
    n_res = len(ctx.levels)
    left, right, gap = 0.06, 0.965, 0.018
    width = (right - left - gap * (n_res - 1)) / n_res
    grow = ramp(p, 0.05, 0.65)
    show_boxes = ramp(p, 0.68, 0.2)
    for r, level in enumerate(ctx.levels):
        labels = ctx.res_labels[r]
        boxes = [b for b in ctx.cluster_boxes[r] if b[2] <= DISPLAY_FMAX]
        ax = fig.add_axes([left + r * (width + gap), 0.09, width, 0.72])
        title = f"M = {level}:  {len(boxes)} clusters" if show_boxes > 0.5 else f"M = {level}"
        style_axes(ax, title, size=9.5)
        energy = log_norm(ctx.res_net[r].image, ctx.net_ref)
        selected = labels.image > 0
        grown = ctx.cluster_depth[r] <= grow
        img = np.where(grown, energy, np.where(selected, 0.1, np.nan))
        show_tf(ax, labels, ctx.tf_cmap, image=img)
        if show_boxes > 0.01:
            placed: list[tuple[float, float]] = []
            total_px = max(sum(b[4] for b in boxes), 1)
            for k, (t0, t1, f0, f1, n_px, _) in enumerate(boxes):
                ax.add_patch(Rectangle((t0, f0), t1 - t0, f1 - f0, fill=False, edgecolor=INK, lw=0.9, alpha=show_boxes))
                if n_px < 0.04 * total_px:
                    continue
                x, y = t0, min(f1 + 8, DISPLAY_FMAX - 30)
                while any(abs(x - px) < 0.45 and abs(y - py) < 28 for px, py in placed):
                    y += 30
                placed.append((x, y))
                ax.text(x, y, f"{n_px} px", color=INK, fontsize=7.5, va="bottom", ha="left", alpha=show_boxes)
        tf_axes(ax, ctx, xlabel=True, ylabel=(r == 0))
    fig.text(
        0.965, 0.855, "each cluster grows from its loudest pixel through 8-neighbours (|Δt| ≤ 1, |Δf| ≤ 1 pixel)",
        color=INK_3, fontsize=9, ha="right", va="center",
    )


# ---------------------------------------------------------------------------
# scene 6: supercluster


def scene_supercluster(fig: plt.Figure, ctx: RenderContext, p: float) -> None:
    n_res = len(ctx.levels)
    show_facets = ramp(p, 0.0, 0.15)
    move = ramp(p, 0.15, 0.35)
    show_links = ramp(p, 0.5, 0.2)
    show_outline = ramp(p, 0.72, 0.2)
    left, right, gap = 0.06, 0.64, 0.012
    width = (right - left - gap * (n_res - 1)) / n_res
    target = np.array([0.06, 0.08, 0.58, 0.47])

    arrived = []
    for r, level in enumerate(ctx.levels):
        start = np.array([left + r * (width + gap), 0.63, width, 0.19])
        ax = fig.add_axes(start)
        style_axes(ax, f"M = {level} fragments", size=9)
        show_tf(ax, ctx.sc_masks[r], ctx.cov_cmap, vmin=0, vmax=1, alpha=show_facets, image=np.where(ctx.sc_masks[r].image, 0.55, np.nan))
        tf_axes(ax, ctx, ylabel=(r == 0))
        stagger = ramp(move, 0.12 * r, 0.52)
        if 0.0 < stagger < 1.0:
            pos = start + (target - start) * stagger
            ghost = fig.add_axes(pos, zorder=20 + r)
            ghost.patch.set_alpha(0.0)
            show_tf(ghost, ctx.sc_masks[r], ctx.cov_cmap, vmin=0, vmax=1, alpha=0.8, image=np.where(ctx.sc_masks[r].image, 0.55, np.nan))
            ghost.set_xlim(0, ctx.t_end)
            ghost.set_ylim(0, DISPLAY_FMAX)
            ghost.axis("off")
            ghost.add_patch(Rectangle((0, 0), 1, 1, transform=ghost.transAxes, fill=False, edgecolor=INK, lw=0.8, alpha=0.8 * (1.0 - stagger)))
        if stagger >= 1.0:
            arrived.append(r)

    ax = fig.add_axes(target)
    style_axes(ax, "merged TF plane  ·  number of resolutions covering each tile", size=9.5)
    if arrived:
        coverage = np.sum([ctx.coverage[r] for r in arrived], axis=0).astype(float)
        img = np.where(coverage > 0, coverage, np.nan)
        ax.imshow(img, extent=[0, ctx.t_end, 0, DISPLAY_FMAX], origin="lower", aspect="auto", interpolation="nearest", cmap=ctx.cov_cmap, vmin=1, vmax=n_res)
    if show_links > 0.01:
        n = int(show_links * len(ctx.sc_links))
        for a, b in ctx.sc_links[:n]:
            (ta, fa), (tb, fb) = ctx.frag_centroids[a], ctx.frag_centroids[b]
            ax.plot([ta, tb], [fa, fb], color=INK, lw=0.6, alpha=0.55)
        pts = np.array(list(ctx.frag_centroids.values()))
        ax.scatter(pts[:, 0], pts[:, 1], s=10, color=INK, alpha=show_links, zorder=5, linewidths=0)
    if show_outline > 0.01:
        cover = np.any(ctx.coverage, axis=0).astype(float)
        cover = ndimage.binary_closing(cover, structure=np.ones((5, 9))).astype(float)
        tc = 0.5 * (ctx.fine_t[:-1] + ctx.fine_t[1:])
        fc = 0.5 * (ctx.fine_f[:-1] + ctx.fine_f[1:])
        ax.contour(tc, fc, cover, levels=[0.5], colors=[INK], linewidths=1.3, alpha=show_outline)
        for t_m, f_m, reason in ctx.rejected:
            ax.scatter([t_m], [f_m], marker="x", s=36, color=INK_3, alpha=show_outline, linewidths=1.2)
            ax.text(t_m + 0.02, f_m + 14, f"rejected: {reason}", color=INK_3, fontsize=7, alpha=show_outline)
    tf_axes(ax, ctx, xlabel=True)

    # legend for coverage count (sequential ramp)
    lax = fig.add_axes([0.66, 0.66, 0.30, 0.03])
    lax.imshow(np.arange(1, n_res + 1)[None, :], cmap=ctx.cov_cmap, vmin=1, vmax=n_res, aspect="auto", extent=[0.5, n_res + 0.5, 0, 1])
    lax.set_yticks([])
    lax.set_xticks(np.arange(1, n_res + 1))
    lax.tick_params(colors="#a8b3c2", labelsize=8, length=0)
    for spine in lax.spines.values():
        spine.set_visible(False)
    fig.text(0.66, 0.705, "resolutions covering a tile", color=INK_3, fontsize=8.5)

    ax = fig.add_axes([0.66, 0.42, 0.30, 0.2])
    card(ax, show_links)
    lines = [
        ("fragments", f"{len(ctx.sc.fragments)} from {len(np.unique(ctx.sc.res))} resolutions"),
        ("TF-gap links", f"{len(ctx.sc_links)}  (gap ≤ {ctx.cfg.tf_gap:g} px)"),
        ("defragment", f"Tgap {ctx.cfg.t_gap:g} s, Fgap {ctx.cfg.f_gap:g} Hz"),
        ("pixels", f"{ctx.sc.res.size}"),
    ]
    text_lines(ax, lines, 0.05, 0.8, 0.2, show_links, size=9.5)

    ax = fig.add_axes([0.66, 0.16, 0.30, 0.19])
    style_axes(ax, "sub-network check: excess energy per detector", size=9.5)
    excess = ctx.sc.det_excess
    ypos = np.arange(ctx.n_det)[::-1]
    if show_outline > 0.01:
        ax.barh(ypos, excess * show_outline, color=ctx.colors, height=0.55)
        for y, d in zip(ypos, range(ctx.n_det)):
            ax.text(excess[d] * show_outline, y, f"  {excess[d]:,.0f}", color=INK_2, fontsize=8.5, va="center")
    ax.set_yticks(ypos)
    ax.set_yticklabels(ctx.ifos)
    ax.set_xticks([])
    ax.set_xlim(0, float(np.max(excess)) * 1.4)
    ax.set_ylim(-0.6, ctx.n_det - 0.4)
    ax.tick_params(axis="y", labelsize=9)
    verdict = "passes" if ctx.sc.passed else "fails"
    fig.text(
        0.66, 0.11, rf"subnet $=1-\max_d E_d\,/\sum_d E_d={ctx.sc.subnet:.2f}$",
        color=INK, fontsize=9.5, alpha=show_outline,
    )
    fig.text(0.66, 0.07, f"{verdict} the cut ≥ {ctx.cfg.subnet_cut:g} (a single-detector glitch gives ≈ 0)", color=INK_3, fontsize=8.5, alpha=show_outline)


# ---------------------------------------------------------------------------
# scene 7: sky loop


def draw_sky_grid(ax: plt.Axes, alpha: float = 1.0) -> None:
    for ra_h in range(0, 24, 3):
        lat = np.linspace(-np.pi / 2, np.pi / 2, 90)
        x, y = mollweide_forward(np.full_like(lat, float(ra_to_lon(ra_h / 24 * 2 * np.pi))), lat)
        ax.plot(x, y, color="#2a3a4d", lw=0.5, alpha=alpha)
        if 0 < ra_h:
            xl, yl = mollweide_forward(np.array([float(ra_to_lon(ra_h / 24 * 2 * np.pi))]), np.array([0.0]))
            ax.text(xl[0], yl[0] - 0.08, f"{ra_h}h", color=INK_3, fontsize=7, ha="center", va="top", alpha=alpha)
    for dec_deg in (-60, -30, 0, 30, 60):
        lon = np.linspace(-np.pi, np.pi, 180)
        x, y = mollweide_forward(lon, np.full_like(lon, np.radians(dec_deg)))
        ax.plot(x, y, color="#2a3a4d", lw=0.5, alpha=alpha)
        xl, yl = mollweide_forward(np.array([np.pi]), np.array([np.radians(dec_deg)]))
        ax.text(xl[0] + 0.06, yl[0], f"{dec_deg:+d}°", color=INK_3, fontsize=7, ha="left", va="center", alpha=alpha)
    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(2 * np.sqrt(2) * np.cos(t), np.sqrt(2) * np.sin(t), color="#34475c", lw=1.0)


def scene_sky(fig: plt.Figure, ctx: RenderContext, p: float) -> None:
    scan = ctx.scan
    scan_p = min(p / 0.78, 1.0)
    count = max(1, int(round(scan_p * ctx.n_sky)))
    current = int(ctx.scan_order[min(count, ctx.n_sky) - 1])
    finish = ramp(p, 0.8, 0.15)

    ax = fig.add_axes([0.02, 0.10, 0.60, 0.74])
    ax.set_facecolor(BG)
    ax.axis("off")
    revealed = ctx.rank[ctx.sky_idx] < count
    values = ctx.stat_norm[ctx.sky_idx]
    rgba = ctx.sky_cmap(values)
    rgba[~revealed] = matplotlib.colors.to_rgba("#0a0f16")
    rgba[~ctx.sky_inside] = (0, 0, 0, 0)
    ax.imshow(rgba, extent=ctx.sky_extent, origin="lower", interpolation="nearest")
    draw_sky_grid(ax)
    ax.set_xlim(-2 * np.sqrt(2) * 1.02, 2 * np.sqrt(2) * 1.08)
    ax.set_ylim(-np.sqrt(2) * 1.08, np.sqrt(2) * 1.12)
    ax.set_aspect("equal")
    ax.text(0.0, 1.0, f"HEALPix nside = {ctx.cfg.nside}  ({ctx.n_sky} directions, RING order)   ·   colour: sky statistic $L_o\\cdot cc$ / max",
            color=INK_2, fontsize=9, transform=ax.transAxes, va="bottom")
    if finish < 0.999:
        cx, cy = ctx.sky_xy[0][current], ctx.sky_xy[1][current]
        ax.scatter([cx], [cy], s=90, facecolors="none", edgecolors=INK, linewidths=1.3, zorder=6, alpha=1 - finish)
    if finish > 0.01:
        X = np.linspace(ctx.sky_extent[0], ctx.sky_extent[1], ctx.sky_idx.shape[1])
        Y = np.linspace(ctx.sky_extent[2], ctx.sky_extent[3], ctx.sky_idx.shape[0])
        for d, img in enumerate(ctx.delay_imgs, start=1):
            ax.contour(X, Y, img, levels=[float(scan.rel_delay[d][scan.best])], colors=[ctx.colors[d]], linewidths=1.0, linestyles="--", alpha=0.9 * finish)
        ax.scatter([ctx.best_xy[0]], [ctx.best_xy[1]], marker="*", s=260, c=ACCENT, edgecolors="#ffffff", linewidths=0.9, zorder=8, alpha=finish)
        ax.scatter([ctx.src_xy[0]], [ctx.src_xy[1]], marker="o", s=110, facecolors="none", edgecolors="#ffffff", linewidths=1.5, zorder=8, alpha=finish)
        close = np.hypot(ctx.best_xy[0] - ctx.src_xy[0], ctx.best_xy[1] - ctx.src_xy[1]) < 0.25
        if close:
            ax.text(ctx.best_xy[0], ctx.best_xy[1] + 0.2, "max statistic at the true source", color=ACCENT, fontsize=8.5, ha="center", va="bottom", alpha=finish)
        else:
            ax.text(ctx.best_xy[0], ctx.best_xy[1] + 0.2, "max statistic", color=ACCENT, fontsize=8.5, ha="center", va="bottom", alpha=finish)
            ax.text(ctx.src_xy[0], ctx.src_xy[1] - 0.14, "true source", color=INK, fontsize=8.5, ha="center", va="top", alpha=finish)
        rings = ",  ".join(
            f"{ctx.ifos[d]}−{ctx.ifos[0]} {scan.rel_delay[d][scan.best] * 1e3:+.2f} ms" for d in range(1, ctx.n_det)
        )
        note = "a two-detector network localises to a ring" if ctx.n_det == 2 else "the rings cross at the source"
        ax.text(0.02, 0.02, f"dashed: constant delay  {rings}  —  {note}", color=INK_2, fontsize=8.5, transform=ax.transAxes, alpha=finish)

    # per-pixel coherent energy at the current direction
    ax = fig.add_axes([0.655, 0.56, 0.31, 0.26])
    shown = scan.best if finish > 0.5 else current
    style_axes(ax, rf"per-pixel coherent energy $E_c$ (M = {ctx.cfg.display_level})", size=9.5)
    cmap_crop = ctx.coherent_map(shown)
    img = np.where(ctx.sc_display_mask.image, cmap_crop.image, np.nan)
    show_tf(ax, cmap_crop, ctx.div_cmap, vmin=-ctx.pixel_ec_ref, vmax=ctx.pixel_ec_ref, image=img)
    tf_axes(ax, ctx, xlabel=False)
    ax.set_xlim(0.35, 1.65)

    # readout
    ax = fig.add_axes([0.655, 0.30, 0.31, 0.22])
    card(ax)
    idx = shown
    delays = scan.rel_delay[:, idx] * 1e3
    delay_txt = ", ".join(f"{ctx.ifos[d]} {delays[d]:+.2f}" for d in range(1, ctx.n_det)) if ctx.n_det > 1 else "-"
    e_plus, e_cross, g = sm.dpf_basis(scan.fplus[:, idx : idx + 1], scan.fcross[:, idx : idx + 1], ctx.cfg.delta)
    fp_norm = float(np.sqrt(np.sum((scan.fplus[:, idx] ** 2 + scan.fcross[:, idx] ** 2)) / ctx.n_det))
    lines = [
        ("direction", f"#{idx:4d}   α {scan.ra[idx]:.2f}  δ {scan.dec[idx]:+.2f}"),
        (f"Δτ vs {ctx.ifos[0]} (ms)", delay_txt),
        ("cross-pol. weight g", "0  (two-detector hard constraint)" if ctx.n_det == 2 else f"{float(g[0]):.3f}   |F| {fp_norm:.2f}"),
        (r"$L_o$  /  $E_c$  /  null", f"{scan.likelihood[idx] / ctx.norm:7.0f} / {scan.coherent[idx] / ctx.norm:6.0f} / {scan.null[idx] / ctx.norm:5.0f}"),
        ("netcc", f"{scan.net_cc[idx]:.3f}"),
    ]
    text_lines(ax, lines, 0.04, 0.85, 0.175, 1.0, size=9)

    # statistic trace along the scan
    ax = fig.add_axes([0.655, 0.08, 0.31, 0.16])
    style_axes(ax, None)
    order = ctx.scan_order[:count]
    ax.plot(np.arange(count), ctx.stat_norm[order], color="#6FA8DC", lw=0.6)
    ax.plot(np.arange(count), np.maximum.accumulate(ctx.stat_norm[order]), color=ACCENT, lw=1.1)
    ax.set_xlim(0, ctx.n_sky)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("sky index in scan", fontsize=8)
    ax.set_ylabel("stat / max", fontsize=8)
    ax.grid(color=GRID, lw=0.4)
    ax.text(0.01, 0.95, "running maximum", color=ACCENT, fontsize=7.5, transform=ax.transAxes, va="top")


# ---------------------------------------------------------------------------
# scene 8: reconstruction


def scene_reconstruction(fig: plt.Figure, ctx: RenderContext, p: float) -> None:
    draw = ramp(p, 0.05, 0.6)
    show_tf_p = ramp(p, 0.3, 0.3)
    show_null = ramp(p, 0.55, 0.3)
    t = ctx.t
    lo, hi = 0.42, 1.58
    rows = ctx.n_det
    top, bottom, gap = 0.84, 0.09, 0.07
    height = (top - bottom - gap * (rows - 1)) / rows
    scale = max(float(np.max(np.abs(ctx.inj))), 1e-12)
    n_draw = int(draw * t.size)
    for d in range(rows):
        ax = fig.add_axes([0.06, top - (d + 1) * height - d * gap, 0.56, height])
        style_axes(ax, f"{ctx.ifos[d]} whitened data, injection and reconstruction  ·  overlap {ctx.overlap[d]:.3f}", size=9.5)
        ax.plot(t, ctx.white[d] / scale, color="#5b6878", lw=0.5, label="whitened data")
        ax.plot(t[:n_draw], ctx.recon[d, :n_draw] / scale, color=ACCENT, lw=1.6, alpha=0.95, label="reconstructed")
        ax.plot(t, ctx.inj[d] / scale, color=INK, lw=0.55, label="injected (whitened)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(-1.35, 1.35)
        ax.set_yticks([])
        ax.grid(color=GRID, lw=0.4)
        if d == 0:
            ax.legend(loc="upper right", fontsize=7.5, frameon=False, labelcolor=INK_2, ncol=3)
        if d == rows - 1:
            ax.set_xlabel("time from GPS reference (s)", fontsize=8)
        else:
            ax.set_xticklabels([])

    ref = float(np.percentile(ctx.recon_tf.image[ctx.recon_tf.image > 0], 99)) if np.any(ctx.recon_tf.image > 0) else 1.0
    ax = fig.add_axes([0.67, 0.50, 0.29, 0.30])
    style_axes(ax, r"signal $s=P\,x$  at the best sky point", size=9.5)
    show_tf(ax, ctx.recon_tf, ctx.tf_cmap, alpha=show_tf_p, image=np.where(ctx.recon_tf.image > 0, log_norm(ctx.recon_tf.image, ref), np.nan))
    tf_axes(ax, ctx)
    ax.set_xlim(lo, hi)

    ax = fig.add_axes([0.67, 0.09, 0.29, 0.30])
    style_axes(ax, r"null stream $x-s$  (same colour scale)", size=9.5)
    show_tf(ax, ctx.null_tf, ctx.tf_cmap, alpha=show_null, image=np.where(ctx.sc_display_mask.image, log_norm(ctx.null_tf.image, ref), np.nan))
    tf_axes(ax, ctx, xlabel=True)
    ax.set_xlim(lo, hi)
    ev = ctx.event
    fig.text(
        0.67, 0.435, rf"$E_c={ev.coherent:.0f}$,  null $N={ev.null:.0f}$,  $cc=E_c/(|E_c|+N)={ev.net_cc:.3f}$",
        color=INK, fontsize=9.5, alpha=show_null,
    )


# ---------------------------------------------------------------------------
# scene 9: background


def scene_background(fig: plt.Figure, ctx: RenderContext, p: float) -> None:
    lag_p = min(p / 0.7, 1.0)
    n_lags = len(ctx.r.lags)
    done = int(round(lag_p * n_lags))
    show_card = ramp(p, 0.68, 0.22)
    lag_value = float(ctx.r.lags[done - 1]) if done > 0 else 0.0

    total = ctx.full_t[-1]
    strip_gap = 0.045
    strip_h = (0.24 - strip_gap * (ctx.n_det - 1)) / ctx.n_det
    for d in range(ctx.n_det):
        ax = fig.add_axes([0.06, 0.84 - (d + 1) * strip_h - d * strip_gap, 0.56, strip_h])
        shift = int(round(d * lag_value * ctx.fs / 4))
        series = np.roll(ctx.full_white[d], shift)
        signal = np.roll(ctx.full_signal[d], shift)
        style_axes(ax, (f"{ctx.ifos[d]}" + (f" shifted by {d * lag_value:.2f} s (circular)" if d > 0 else " fixed")), size=9)
        ax.plot(ctx.full_t, series, color=ctx.colors[d], lw=0.35, alpha=0.8)
        ax.plot(ctx.full_t, signal, color=ACCENT, lw=0.45)
        ax.axvspan(0, ctx.cfg.edge_seconds, color=BG, alpha=0.7)
        ax.axvspan(total - ctx.cfg.edge_seconds, total, color=BG, alpha=0.7)
        ax.set_xlim(0, total)
        ax.set_ylim(-5, 5)
        ax.set_yticks([])
        ax.grid(color=GRID, lw=0.4)
        if d < ctx.n_det - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("segment time (s)", fontsize=8)
    fig.text(0.62, 0.855, f"lag {done} / {n_lags}   ·   yellow: injected signal", color=ACCENT_2, fontsize=9.5, ha="right")

    ax = fig.add_axes([0.06, 0.08, 0.56, 0.44])
    style_axes(ax, r"time-slide triggers: coherent SNR $\rho$ vs network correlation $cc$", size=9.5)
    ax.axhspan(ctx.cfg.net_cc_cut, 1.05, xmin=0, xmax=1, color="#3776AB", alpha=0.06)
    ax.axhline(ctx.cfg.net_cc_cut, color=INK_3, lw=0.8, ls="--")
    ax.axvline(ctx.cfg.net_rho_cut, color=INK_3, lw=0.8, ls="--")
    ax.text(0.72, ctx.cfg.net_cc_cut + 0.02, f"netCC = {ctx.cfg.net_cc_cut:g}", color=INK_3, fontsize=8)
    ax.text(ctx.cfg.net_rho_cut * 1.05, 0.02, f"netRHO = {ctx.cfg.net_rho_cut:g}", color=INK_3, fontsize=8)
    pts = [trig for lag in ctx.lag_triggers[:done] for trig in lag]
    if pts:
        arr = np.array(pts, dtype=float)
        passed = arr[:, 2] > 0.5
        ax.scatter(arr[~passed, 0], arr[~passed, 1], s=30, facecolors="none", edgecolors="#8795a6", linewidths=1.0, label="background, rejected")
        if np.any(passed):
            ax.scatter(arr[passed, 0], arr[passed, 1], s=30, color="#3987e5", edgecolors=BG, linewidths=0.8, label="background, passes cuts")
    ev = ctx.event
    ax.scatter([ev.rho], [ev.net_cc], marker="*", s=300, color=ACCENT, edgecolors="#ffffff", linewidths=0.9, zorder=6, label="zero-lag event")
    ax.set_xscale("log")
    ax.set_xlim(0.7, max(ev.rho * 1.8, 20))
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"$\rho=\sqrt{E_c\,cc/(n_{\rm IFO}-1)}$", fontsize=9)
    ax.set_ylabel("netcc", fontsize=8)
    ax.grid(color=GRID, lw=0.4)
    ax.legend(loc="upper center", fontsize=7.5, frameon=False, labelcolor=INK_2, markerscale=0.7)

    ax = fig.add_axes([0.67, 0.08, 0.29, 0.76])
    card(ax, show_card, edge=ACCENT_2 if show_card > 0.5 else SPINE)
    gps = ctx.cfg.gps_time + ev.time - ctx.window[0]
    far = (
        f"< 1 / {ctx.livetime:.0f} s" if ctx.n_louder == 0 else f"{ctx.n_louder / ctx.livetime:.3g} Hz"
    )
    true_delay = (ctx.r.arrival_delay - ctx.r.arrival_delay[0]) * 1e3
    ax.text(0.06, 0.95, "candidate event", color=ACCENT, fontsize=13, fontweight="bold", transform=ax.transAxes, va="top", alpha=show_card)
    lines = [
        ("GPS time", f"{gps:.3f}"),
        ("frequency band", f"{ev.f_low:.0f} – {ev.f_high:.0f} Hz"),
        ("coherent SNR ρ", f"{ev.rho:.1f}"),
        ("netcc", f"{ev.net_cc:.3f}"),
        ("E_c / null", f"{ev.coherent:.0f} / {ev.null:.0f}"),
        *[
            (f"Δτ {ctx.ifos[d]}−{ctx.ifos[0]}", f"{ev.rel_delay[d] * 1e3:+.2f} ms  (true {true_delay[d]:+.2f})")
            for d in range(1, ctx.n_det)
        ],
        ("best sky (α, δ)", f"{ev.ra:.2f}, {ev.dec:+.2f} rad"),
        ("cuts", ("passed" if ev.passed else "failed") + f"  (ρ ≥ {ctx.cfg.net_rho_cut:g}, cc ≥ {ctx.cfg.net_cc_cut:g})"),
        ("time slides", f"{n_lags} lags × {ctx.livetime / max(n_lags, 1):.0f} s"),
        ("background", f"{ctx.n_background} triggers, {ctx.n_passed_bg} pass cuts"),
        ("louder than event", f"{ctx.n_louder}"),
        ("FAR", far),
    ]
    text_lines(ax, lines, 0.06, 0.85, 0.062, show_card, size=9.5)


SCENE_FUNCS = {
    "projection": scene_projection,
    "whitening": scene_whitening,
    "wdm": scene_wdm,
    "selection": scene_selection,
    "clustering": scene_clustering,
    "supercluster": scene_supercluster,
    "sky": scene_sky,
    "reconstruction": scene_reconstruction,
    "background": scene_background,
}


# ---------------------------------------------------------------------------
# timeline and rendering


@dataclass
class Timeline:
    names: list[str]
    starts: list[int]
    lengths: list[int]

    @property
    def total(self) -> int:
        return self.starts[-1] + self.lengths[-1]

    def locate(self, frame: int) -> tuple[str, float, int, int]:
        for name, start, length in zip(self.names, self.starts, self.lengths):
            if frame < start + length:
                local = frame - start
                return name, local / max(length - 1, 1), local, length
        name, start, length = self.names[-1], self.starts[-1], self.lengths[-1]
        return name, 1.0, length - 1, length


def build_timeline(args: argparse.Namespace) -> Timeline:
    chosen = [s for s in SCENES if s[0] in args.scenes]
    weight = sum(s[2] for s in chosen)
    total = int(round(args.duration * args.fps))
    lengths = [max(2, int(round(total * s[2] / weight))) for s in chosen]
    starts = list(np.cumsum([0] + lengths[:-1]).astype(int))
    return Timeline([s[0] for s in chosen], starts, lengths)


def draw_scene(fig: plt.Figure, ctx: RenderContext, name: str, progress: float, fade: float = 0.0) -> None:
    fig.clear()
    fig.patch.set_facecolor(BG)
    index = [s[0] for s in SCENES].index(name)
    draw_header(fig, index, SCENES[index][1])
    SCENE_FUNCS[name](fig, ctx, progress)
    draw_fade(fig, fade)


def fade_for(local: int, length: int, fps: int) -> float:
    n = max(1, int(round(0.3 * fps)))
    if local < n:
        return 1.0 - local / n
    if local >= length - max(1, n // 2):
        return (local - (length - max(1, n // 2)) + 1) / max(1, n // 2) * 0.85
    return 0.0


def render_mp4(ctx: RenderContext, args: argparse.Namespace, out_dir: Path) -> Path:
    timeline = build_timeline(args)
    frame_count = timeline.total if args.frames_limit is None else min(args.frames_limit, timeline.total)
    output = out_dir / "pycwb_search_animation.mp4"
    fig = plt.figure(figsize=(args.width / args.dpi, args.height / args.dpi), dpi=args.dpi)
    writer = FFMpegWriter(
        fps=args.fps, codec="libx264", bitrate=4500,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        metadata={"title": "pycWB coherent search animation"},
    )
    print(f"Rendering {frame_count} frame(s) to {output}")
    step = max(1, frame_count // 20)
    start = timer.perf_counter()
    with writer.saving(fig, str(output), dpi=args.dpi):
        for frame in range(frame_count):
            name, progress, local, length = timeline.locate(frame)
            draw_scene(fig, ctx, name, progress, fade_for(local, length, args.fps))
            writer.grab_frame(facecolor=fig.get_facecolor())
            if (frame + 1) % step == 0 or frame + 1 == frame_count:
                elapsed = timer.perf_counter() - start
                print(f"  frame {frame + 1}/{frame_count}  ({elapsed:.0f} s, {name})", flush=True)
    plt.close(fig)
    return output


def render_stills(ctx: RenderContext, args: argparse.Namespace, out_dir: Path) -> list[Path]:
    fig = plt.figure(figsize=(args.width / args.dpi, args.height / args.dpi), dpi=args.dpi)
    paths = []
    for name in args.scenes:
        for progress in args.stills:
            draw_scene(fig, ctx, name, float(progress))
            path = out_dir / f"still_{[s[0] for s in SCENES].index(name) + 1}_{name}_{int(round(progress * 100)):03d}.png"
            fig.savefig(path, dpi=args.dpi, facecolor=fig.get_facecolor())
            paths.append(path)
    plt.close(fig)
    return paths


def render_gif(mp4_path: Path, args: argparse.Namespace, out_dir: Path) -> Path:
    gif_path = out_dir / "pycwb_search_animation.gif"
    palette_path = out_dir / "palette.png"
    scale_filter = f"fps={args.gif_fps},scale={args.gif_width}:-1:flags=lanczos"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(mp4_path), "-vf", f"{scale_filter},palettegen=stats_mode=diff",
         "-frames:v", "1", "-update", "1", str(palette_path)],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(mp4_path), "-i", str(palette_path), "-filter_complex",
         f"{scale_filter}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", str(gif_path)],
        check=True,
    )
    palette_path.unlink(missing_ok=True)
    return gif_path


# ---------------------------------------------------------------------------
# data products


def save_data(ctx: RenderContext, out_dir: Path, args: argparse.Namespace) -> None:
    r = ctx.r
    zero = r.zero_lag
    arrays = {
        "time": ctx.t,
        "hplus": r.hplus,
        "clean": r.clean[:, ctx.i0 : ctx.i1],
        "raw": r.raw[:, ctx.i0 : ctx.i1],
        "whitened": r.whitened[:, ctx.i0 : ctx.i1],
        "injected_whitened": ctx.inj,
        "reconstructed_whitened": ctx.recon,
        "noise_rms_freq": r.white_freq,
        "noise_rms": r.white_rms,
        "sky_ra": ctx.scan.ra,
        "sky_dec": ctx.scan.dec,
        "sky_statistic": ctx.scan.statistic,
        "sky_net_cc": ctx.scan.net_cc,
        "sky_coherent": ctx.scan.coherent,
        "sky_null": ctx.scan.null,
        "sky_rel_delay": ctx.scan.rel_delay,
        "lags": r.lags,
        "background_rho": np.array([t.rho for lag in r.background for t in lag]),
        "background_cc": np.array([t.net_cc for lag in r.background for t in lag]),
        "background_lag": np.array([lag_value for lag_value, lag in zip(r.lags, r.background) for _ in lag]),
    }
    for i, res in enumerate(zero.resolutions):
        key = f"M{res.level}"
        arrays[f"{key}_energy"] = sm.pixel_energy(crop_tf(res.coeff, res.dt, res.df, ctx.window).image)
        arrays[f"{key}_net_energy"] = ctx.res_net[i].image
        arrays[f"{key}_selected"] = ctx.res_selected[i].image
        arrays[f"{key}_labels"] = ctx.res_labels[i].image
        arrays[f"{key}_extent"] = np.array(ctx.res_net[i].extent)
    np.savez_compressed(out_dir / "animation_data.npz", **arrays)

    event = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in asdict(r.event).items()}
    event["gps_time"] = r.config.gps_time + r.event.time - r.window[0]
    event["time"] = r.event.time - r.window[0]
    cfg = asdict(r.config)
    cfg["ifos"] = list(cfg["ifos"])
    cfg["levels"] = list(cfg["levels"])
    metadata = {
        "input": str(args.input),
        "config": cfg,
        "sample_rate": r.sample_rate,
        "source": {
            "ra_rad": r.config.source_ra, "dec_rad": r.config.source_dec,
            "antenna": {ifo: {"fplus": float(r.antenna[d][0]), "fcross": float(r.antenna[d][1])} for d, ifo in enumerate(ctx.ifos)},
            "arrival_delay_s": {ifo: float(r.arrival_delay[d]) for d, ifo in enumerate(ctx.ifos)},
        },
        "resolutions": [
            {"M": res.level, "dt": res.dt, "df": res.df, "threshold": res.threshold,
             "selected_pixels": int(res.selected.sum()), "clusters": int(res.labels.max())}
            for res in zero.resolutions
        ],
        "supercluster": {"pixels": int(ctx.sc.res.size), "fragments": len(ctx.sc.fragments), "subnet": ctx.sc.subnet},
        "event": event,
        "reconstruction_overlap": {ifo: ctx.overlap[d] for d, ifo in enumerate(ctx.ifos)},
        "background": {
            "lags": len(r.lags), "livetime_s": ctx.livetime, "triggers": ctx.n_background,
            "passing_cuts": ctx.n_passed_bg, "louder_than_event": ctx.n_louder,
        },
        "scenes": [s[0] for s in SCENES if s[0] in args.scenes],
        "duration": args.duration,
        "fps": args.fps,
        "didactic_note": (
            "All panels are computed from the simulated data: PyCBC projection, WDM transforms, "
            "per-layer whitening, delay-maximised pixel selection, clustering, supercluster linking, "
            "a HEALPix sky loop with DPF + regulator, reconstruction and time slides. The likelihood is a "
            "compact version of the cWB 2G statistic, not the production likelihoodWP module."
        ),
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def dependency_smoke_check(formats: list[str]) -> None:
    required = ["numpy", "matplotlib", "scipy", "pycbc", "wdm_wavelet", "healpy"]
    missing = []
    for module in required:
        try:
            __import__(module)
        except Exception:
            missing.append(module)
    if missing:
        raise RuntimeError(f"Missing required module(s): {', '.join(missing)}")
    if formats and shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on PATH")


def main() -> None:
    args = parse_args()
    dependency_smoke_check(args.format)
    cfg = sm.SearchConfig(
        ifos=tuple(args.ifos), levels=tuple(args.levels), noise_sigma=args.noise_sigma,
        noise_seed=args.noise_seed, bpp=args.bpp, n_lags=args.lags, lag_step=args.lag_step,
        nside=args.nside or (16 if len(args.ifos) == 2 else 64),
    )
    print("Running the coherent search model")
    start = timer.perf_counter()
    result = sm.run_search(args.input, cfg)
    print(f"  done in {timer.perf_counter() - start:.0f} s: rho = {result.event.rho:.1f}, netcc = {result.event.net_cc:.3f}")
    matplotlib.rcParams.update(RC_STYLE)
    ctx = RenderContext(result, args)
    args.out.mkdir(parents=True, exist_ok=True)
    save_data(ctx, args.out, args)
    print(f"Saved data to {args.out / 'animation_data.npz'} and {args.out / 'metadata.json'}")

    if args.stills:
        paths = render_stills(ctx, args, args.out)
        print(f"Saved {len(paths)} still(s) to {args.out}")
    if args.format:
        mp4_path = render_mp4(ctx, args, args.out)
        print(f"Saved MP4 to {mp4_path}")
        if "gif" in args.format:
            gif_path = render_gif(mp4_path, args, args.out)
            print(f"Saved GIF to {gif_path}")


if __name__ == "__main__":
    main()
