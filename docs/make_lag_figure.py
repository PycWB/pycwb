#!/usr/bin/env python
"""Write source/_static/diagrams/lags_superlags.svg (animated, CSS only).

The figure illustrates the two time-shift levels used for background
estimation, following the code:

* lags  (``WaveSegment.lag_shifts``, ``coherence_native.selection``): inside a
  job segment the first detector in ``ifo`` is read at t + m * lagStep; the
  shift wraps around the analysis window (circular), other detectors stay.
* superlags (``superlag.generate_slags``, ``WaveSegment.physical_analyze_starts``):
  detector k reads GPS time T - s_k * segLen, pairing different segments. There
  is no wrap-around; only time where both detectors have data becomes a job.

Usage: python docs/make_lag_figure.py   (standard library only)
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent / "source" / "_static" / "diagrams" / "lags_superlags.svg"

WIDTH, HEIGHT = 700, 560
INK, INK_2, EDGE = "#263238", "#546e7a", "#455a64"
H1_FILL, H1_STROKE = "#fde8e8", "#d65151"
L1_FILL, L1_STROKE = "#e1ecfb", "#2f78d0"
STAR_FILL, STAR_STROKE = "#ffd43b", "#9a7400"
GOOD = "#2e7d32"
MUTED = "#90a4ae"
LOOP = 12  # seconds for both panels

# panel (a): one job segment split into chunks
A_X0, A_CHUNK, A_N = 118, 66, 8
A_H1_Y, A_L1_Y, A_ROW_H = 96, 156, 34
A_EVENT = 2  # zero-based chunk holding the GW signal
A_LAGS = 4

# panel (b): consecutive job segments of the run
B_X0, B_SEG_W, B_GAP, B_N = 118, 88, 12, 5
B_PITCH = B_SEG_W + B_GAP
B_H1_Y, B_L1_Y, B_ROW_H = 350, 410, 34
B_EVENT = 2
B_SLAGS = 3


def star(cx: float, cy: float, r: float = 8.0) -> str:
    import math

    pts = []
    for i in range(10):
        rad = r if i % 2 == 0 else r * 0.45
        ang = -math.pi / 2 + i * math.pi / 5
        pts.append(f"{cx + rad * math.cos(ang):.1f},{cy + rad * math.sin(ang):.1f}")
    return f'<polygon points="{" ".join(pts)}" fill="{STAR_FILL}" stroke="{STAR_STROKE}" stroke-width="1"/>'


def text(x, y, s, size=12, color=INK, weight="normal", anchor="start", cls="", extra=""):
    c = f' class="{cls}"' if cls else ""
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" '
        f'text-anchor="{anchor}"{c}{extra}>{s}</text>'
    )


def phase_keyframes(name: str, states: int) -> str:
    """Opacity keyframes: visible for the first 1/states of the loop."""
    share = 100.0 / states
    return (
        f"@keyframes {name} {{ 0% {{opacity:1}} {share - 2:.1f}% {{opacity:1}} "
        f"{share:.1f}% {{opacity:0}} {100 - 1.5:.1f}% {{opacity:0}} 100% {{opacity:1}} }}"
    )


def phase_delay(k: int, states: int) -> str:
    step = LOOP / states
    return "0s" if k == 0 else f"-{LOOP - k * step:.2f}s"


def shift_keyframes(name: str, states: int, dx: float) -> str:
    """Step through translateX(k * dx) with short eased moves."""
    share = 100.0 / states
    frames = []
    for k in range(states):
        start = k * share
        hold = start + share - 4
        frames.append(f"{start + (4 if k else 0):.1f}% {{transform:translateX({k * dx:.1f}px)}}")
        frames.append(f"{hold:.1f}% {{transform:translateX({k * dx:.1f}px)}}")
    frames.append("100% {transform:translateX(0px)}")
    return f"@keyframes {name} {{ {' '.join(frames)} }}"


def panel_a() -> list[str]:
    out = []
    width = A_CHUNK * A_N
    out.append(text(24, 36, "(a) Lags: circular time slides inside one job segment", 15, weight="bold"))
    out.append(text(24, 56, "The first detector in ifo is read at t + m × lagStep; the other detectors stay fixed.", 12, INK_2))

    # wrap-around arrow above the H1 strip
    y = A_H1_Y - 6
    out.append(
        f'<path d="M {A_X0 + 6} {y} C {A_X0 + 6} {y - 22}, {A_X0 + width - 6} {y - 22}, {A_X0 + width - 6} {y}" '
        f'fill="none" stroke="{MUTED}" stroke-width="1.4" stroke-dasharray="4 3" marker-end="url(#arrow)"/>'
    )
    out.append(text(A_X0 + width / 2, y - 20, "data leaving the start re-enter at the end (circular)", 11, INK_2, anchor="middle"))

    # row labels
    out.append(text(24, A_H1_Y + 16, "H1", 14, H1_STROKE, "bold"))
    out.append(text(24, A_H1_Y + 30, "shifted", 11, INK_2))
    out.append(text(24, A_L1_Y + 16, "L1", 14, L1_STROKE, "bold"))
    out.append(text(24, A_L1_Y + 30, "fixed", 11, INK_2))

    # coincidence column around the signal position in the fixed detector
    col_x = A_X0 + A_EVENT * A_CHUNK
    out.append(
        f'<rect class="coinc" x="{col_x - 3}" y="{A_H1_Y - 3}" width="{A_CHUNK + 6}" height="{A_L1_Y - A_H1_Y + A_ROW_H + 6}" '
        f'rx="5" fill="none" stroke="{GOOD}" stroke-width="2" stroke-dasharray="5 3"/>'
    )

    # H1 strip: two copies inside a clip so the leftward shift wraps around
    out.append(f'<g clip-path="url(#clipA)"><g class="h1lag">')
    for copy in range(2):
        for i in range(A_N):
            x = A_X0 + copy * width + i * A_CHUNK
            out.append(
                f'<rect x="{x}" y="{A_H1_Y}" width="{A_CHUNK}" height="{A_ROW_H}" fill="{H1_FILL}" '
                f'stroke="{H1_STROKE}" stroke-width="1.2"/>'
            )
            out.append(text(x + 10, A_H1_Y + 22, f"t{i + 1}", 11, INK))
            if i == A_EVENT:
                out.append(star(x + A_CHUNK - 16, A_H1_Y + A_ROW_H / 2))
    out.append("</g></g>")
    out.append(
        f'<rect x="{A_X0}" y="{A_H1_Y}" width="{width}" height="{A_ROW_H}" fill="none" stroke="{H1_STROKE}" stroke-width="2"/>'
    )

    # L1 strip (fixed)
    for i in range(A_N):
        x = A_X0 + i * A_CHUNK
        out.append(
            f'<rect x="{x}" y="{A_L1_Y}" width="{A_CHUNK}" height="{A_ROW_H}" fill="{L1_FILL}" '
            f'stroke="{L1_STROKE}" stroke-width="1.2"/>'
        )
        out.append(text(x + 10, A_L1_Y + 22, f"t{i + 1}", 11, INK))
        if i == A_EVENT:
            out.append(star(x + A_CHUNK - 16, A_L1_Y + A_ROW_H / 2))
    out.append(
        f'<rect x="{A_X0}" y="{A_L1_Y}" width="{width}" height="{A_ROW_H}" fill="none" stroke="{L1_STROKE}" stroke-width="2"/>'
    )
    out.append(text(A_X0, A_L1_Y + A_ROW_H + 16, "analysis window of one job segment (segEdge padding not shown)", 10.5, INK_2))

    # captions, one per lag
    captions = [
        ("lag 0 (zero lag): the signal is coincident", "a real event appears as a candidate", GOOD),
        ("lag 1: H1 read at t + 1 × lagStep", "the signal no longer lines up: background", INK),
        ("lag 2: H1 read at t + 2 × lagStep", "coincidences are accidental: background", INK),
        ("lag 3: H1 read at t + 3 × lagStep", "each lag reuses the same data: background", INK),
    ]
    ty = A_L1_Y + A_ROW_H + 44
    for k, (title, sub, color) in enumerate(captions):
        delay = phase_delay(k, A_LAGS)
        hidden = "" if k == 0 else ' opacity="0"'
        out.append(
            f'<g class="capA" style="animation-delay:{delay}"{hidden}>'
            + text(A_X0, ty, title, 13, color, "bold")
            + text(A_X0, ty + 18, sub, 12, INK_2)
            + "</g>"
        )
    return out


def panel_b() -> list[str]:
    out = []
    total = B_PITCH * B_N - B_GAP
    top = B_H1_Y - 60
    out.append(f'<line x1="24" y1="{top - 22}" x2="{WIDTH - 24}" y2="{top - 22}" stroke="#cfd8dc" stroke-width="1"/>')
    out.append(text(24, top, "(b) Superlags: pair different segments of the run", 15, weight="bold"))
    out.append(text(24, top + 20, "Detector k reads data from s_k × segLen earlier. Not circular: unpaired time is not analysed.", 12, INK_2))

    out.append(text(24, B_H1_Y + 16, "H1", 14, H1_STROKE, "bold"))
    out.append(text(24, B_H1_Y + 30, "s = 0", 11, INK_2))
    out.append(text(24, B_L1_Y + 16, "L1", 14, L1_STROKE, "bold"))
    out.append(text(24, B_L1_Y + 30, "shift s", 11, INK_2))

    for i in range(B_N):
        x = B_X0 + i * B_PITCH
        out.append(
            f'<rect x="{x}" y="{B_H1_Y}" width="{B_SEG_W}" height="{B_ROW_H}" rx="3" fill="{H1_FILL}" '
            f'stroke="{H1_STROKE}" stroke-width="1.5"/>'
        )
        out.append(text(x + 10, B_H1_Y + 22, f"S{i + 1}", 12, INK))
        if i == B_EVENT:
            out.append(star(x + B_SEG_W - 16, B_H1_Y + B_ROW_H / 2))

    # L1 segments move right by s segments; links show which H1 segment they pair with
    out.append('<g clip-path="url(#clipB)"><g class="l1slag">')
    for i in range(B_N):
        x = B_X0 + i * B_PITCH
        out.append(
            f'<line x1="{x + B_SEG_W / 2}" y1="{B_H1_Y + B_ROW_H + 2}" x2="{x + B_SEG_W / 2}" y2="{B_L1_Y - 2}" '
            f'stroke="{MUTED}" stroke-width="1.4" stroke-dasharray="3 3"/>'
        )
        out.append(
            f'<rect x="{x}" y="{B_L1_Y}" width="{B_SEG_W}" height="{B_ROW_H}" rx="3" fill="{L1_FILL}" '
            f'stroke="{L1_STROKE}" stroke-width="1.5"/>'
        )
        out.append(text(x + 10, B_L1_Y + 22, f"S{i + 1}", 12, INK))
        if i == B_EVENT:
            out.append(star(x + B_SEG_W - 16, B_L1_Y + B_ROW_H / 2))
    out.append("</g></g>")

    # beyond the run: L1 segments shifted past the last H1 segment have no partner
    end = B_X0 + total
    out.append(
        f'<rect x="{end + B_GAP / 2}" y="{B_H1_Y - 6}" width="{WIDTH - end - B_GAP / 2 - 10}" height="{B_L1_Y - B_H1_Y + B_ROW_H + 12}" '
        f'fill="#ffffff" opacity="0.72"/>'
    )
    out.append(text(end + 16, B_H1_Y + 20, "end of", 11, INK_2))
    out.append(text(end + 16, B_H1_Y + 34, "the run", 11, INK_2))

    # H1 segments without an L1 partner for s = 1, 2 are faded
    for i in range(B_SLAGS - 1):
        x = B_X0 + i * B_PITCH
        out.append(
            f'<rect class="nopair{i + 1}" x="{x - 2}" y="{B_H1_Y - 2}" width="{B_SEG_W + 4}" height="{B_ROW_H + 4}" '
            f'fill="#ffffff" opacity="0"/>'
        )

    captions = [
        ("superlag 0: the same segments are paired", "lag 0 here is the physical zero lag; the other lags are background", GOOD),
        ("superlag s = 1: H1 segment j + 1 with L1 segment j", "a new set of job segments; all of their lags are background", INK),
        ("superlag s = 2: H1 segment j + 2 with L1 segment j", "H1 S1–S2 and L1 S4–S5 have no partner and are not analysed", INK),
    ]
    ty = B_L1_Y + B_ROW_H + 30
    for k, (title, sub, color) in enumerate(captions):
        delay = phase_delay(k, B_SLAGS)
        hidden = "" if k == 0 else ' opacity="0"'
        out.append(
            f'<g class="capB" style="animation-delay:{delay}"{hidden}>'
            + text(B_X0, ty, title, 13, color, "bold")
            + text(B_X0, ty + 18, sub, 12, INK_2)
            + "</g>"
        )
    return out


def build() -> str:
    share_b = 100.0 / B_SLAGS
    css = "\n".join(
        [
            "text { font-family: Helvetica, Arial, sans-serif; }",
            shift_keyframes("lagshift", A_LAGS, -A_CHUNK),
            shift_keyframes("slagshift", B_SLAGS, B_PITCH),
            phase_keyframes("showA", A_LAGS),
            phase_keyframes("showB", B_SLAGS),
            # coincidence box is green only at lag 0
            f"@keyframes coinc {{ 0% {{stroke:{GOOD};opacity:1}} 23% {{stroke:{GOOD};opacity:1}} "
            f"25% {{stroke:{MUTED};opacity:0.6}} 98.5% {{stroke:{MUTED};opacity:0.6}} 100% {{stroke:{GOOD};opacity:1}} }}",
            # H1 S1 fades for s >= 1, S2 for s = 2
            f"@keyframes nopair1 {{ 0% {{opacity:0}} {share_b - 2:.1f}% {{opacity:0}} {share_b + 2:.1f}% {{opacity:0.7}} "
            f"98% {{opacity:0.7}} 100% {{opacity:0}} }}",
            f"@keyframes nopair2 {{ 0% {{opacity:0}} {2 * share_b - 2:.1f}% {{opacity:0}} {2 * share_b + 2:.1f}% {{opacity:0.7}} "
            f"98% {{opacity:0.7}} 100% {{opacity:0}} }}",
            f".h1lag {{ animation: lagshift {LOOP}s ease-in-out infinite; }}",
            f".l1slag {{ animation: slagshift {LOOP}s ease-in-out infinite; }}",
            f".capA {{ animation: showA {LOOP}s linear infinite; }}",
            f".capB {{ animation: showB {LOOP}s linear infinite; }}",
            f".coinc {{ animation: coinc {LOOP}s linear infinite; }}",
            f".nopair1 {{ animation: nopair1 {LOOP}s linear infinite; }}",
            f".nopair2 {{ animation: nopair2 {LOOP}s linear infinite; }}",
            # static fallback: hold the zero-lag frame
            "@media (prefers-reduced-motion: reduce) { .h1lag, .l1slag, .capA, .capB, .coinc, .nopair1, .nopair2 { animation: none; } }",
        ]
    )
    body = panel_a() + panel_b()
    note_y = HEIGHT - 16
    body.append(
        text(
            24, note_y,
            "Background livetime sums the post-CAT2 livetime of every (superlag job, lag) pair except the physical zero lag. Not to scale.",
            10.5, INK_2,
        )
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">
<title id="title">Lags and superlags</title>
<desc id="desc">Animated illustration. (a) Lags: inside one job segment the first detector is read at t plus m times lagStep with circular wrap-around while the others stay fixed; only lag 0 keeps a real signal coincident. (b) Superlags: detector k reads data from s_k times segLen earlier, pairing different segments without wrap-around; unpaired segments are not analysed.</desc>
<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{MUTED}"/></marker>
<clipPath id="clipB"><rect x="12" y="{B_H1_Y - 8}" width="{WIDTH - 24}" height="{B_L1_Y - B_H1_Y + B_ROW_H + 16}"/></clipPath>
<clipPath id="clipA"><rect x="{A_X0}" y="{A_H1_Y - 1}" width="{A_CHUNK * A_N}" height="{A_ROW_H + 2}"/></clipPath>
<style>
{css}
</style>
</defs>
<rect x="1" y="1" width="{WIDTH - 2}" height="{HEIGHT - 2}" rx="8" fill="#ffffff" stroke="{EDGE}" stroke-width="2"/>
{chr(10).join(body)}
</svg>
"""


if __name__ == "__main__":
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT}")
