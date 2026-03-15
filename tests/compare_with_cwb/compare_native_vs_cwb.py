#!/usr/bin/env python3
"""
compare_native_vs_cwb.py
========================
Compare pycWB **native** pipeline output (Parquet catalog + trigger JSON)
with cWB **C++ standalone** output (ROOT supercluster file + CWB log).

Usage (from tests/compare_with_cwb/):
    python compare_native_vs_cwb.py

Data sources
------------
- Native:  catalog/catalog.parquet     → Trigger-level event parameters
           trigger/*/cluster.json      → ClusterMeta (likelihood output)
- CWB:    output/cwb_compare/job1_trail0/tmp/job_*.root → supercluster pixels
          output/cwb_compare/job1_trail0/log/output.log → mchirp_2g + debug lines

Field mapping follows pycwb.modules.catalog.convert_root and
pycwb.types.network_event.Event.output() / output_py().
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Paths (relative to this script's directory)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CATALOG_PARQUET = os.path.join(SCRIPT_DIR, "catalog", "catalog.parquet")
TRIGGER_DIR = os.path.join(SCRIPT_DIR, "trigger")
CWB_ROOT_FILE = os.path.join(
    SCRIPT_DIR,
    "output", "cwb_compare", "job1_trail0", "tmp",
    "job_1126258883_job1_trail0_1_118626.root",
)
CWB_LOG = os.path.join(
    SCRIPT_DIR,
    "output", "cwb_compare", "job1_trail0", "log", "output.log",
)
# The hybrid pycwb log (uses C++ likelihood via ROOT bindings)
HYBRID_LOG = os.path.join(SCRIPT_DIR, "log", "output_hybrid.log")
# The native pycwb log
NATIVE_LOG = os.path.join(SCRIPT_DIR, "log", "output.log")


# ═══════════════════════════════════════════════════════════════════════════
# 1. READ NATIVE OUTPUT (Parquet catalog)
# ═══════════════════════════════════════════════════════════════════════════

def read_catalog(path: str) -> "pd.DataFrame":
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pandas()


# ═══════════════════════════════════════════════════════════════════════════
# 2. READ NATIVE TRIGGER JSONs (ClusterMeta)
# ═══════════════════════════════════════════════════════════════════════════

def read_trigger_jsons(trigger_dir: str) -> list[dict]:
    """Return list of (trigger_name, cluster_meta_dict) from trigger JSON files."""
    results = []
    if not os.path.isdir(trigger_dir):
        return results
    for entry in sorted(os.listdir(trigger_dir)):
        cj = os.path.join(trigger_dir, entry, "cluster.json")
        if os.path.isfile(cj):
            with open(cj) as f:
                data = json.load(f)
            results.append({
                "name": entry,
                "cluster_id": data.get("cluster_id"),
                "n_pixels": len(data.get("pixels", [])),
                "meta": data.get("cluster_meta", {}),
            })
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 3. READ CWB ROOT FILE (supercluster pixel data)
# ═══════════════════════════════════════════════════════════════════════════

def read_cwb_supercluster(root_path: str) -> dict:
    """Read the CWB supercluster TTree and return per-cluster summaries."""
    import uproot

    f = uproot.open(root_path)
    sc_dir = f["supercluster"]

    # Find the TTree
    tree = None
    for name, cls in sc_dir.classnames().items():
        if "TTree" in cls:
            tree = sc_dir[name]
            break
    if tree is None:
        print("  [WARN] No TTree found in supercluster directory")
        return {}

    cid = tree["cid"].array(library="np")
    rate = tree["rate"].array(library="np")
    ctime = tree["ctime"].array(library="np")
    cfreq = tree["cfreq"].array(library="np")

    clusters = {}
    for ci in np.unique(cid):
        mask = cid == ci
        clusters[int(ci)] = {
            "n_pixels": int(mask.sum()),
            "rates": sorted(set(rate[mask].tolist())),
            "ctime": float(ctime[mask][0]),
            "cfreq": float(cfreq[mask][0]),
        }
    return clusters


# ═══════════════════════════════════════════════════════════════════════════
# 4. PARSE CWB LOG for event-level data
# ═══════════════════════════════════════════════════════════════════════════

def parse_cwb_log(log_path: str) -> dict:
    """Extract mchirp_2g lines and C++ debug lines from the CWB output log."""
    result = {"mchirp_2g": [], "debug": {}}
    if not os.path.isfile(log_path):
        return result

    with open(log_path) as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        # mchirp_2g : cid slag mchirp chirpEllip chi2 ? chirp_factor
        m = re.match(
            r"mchirp_2g\s*:\s*(\d+)\s+(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+"
            r"([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)",
            line,
        )
        if m:
            result["mchirp_2g"].append({
                "cluster_id": int(m.group(1)),
                "slag": int(m.group(2)),
                "mchirp": float(m.group(3)),
                "chirp_ellip": float(m.group(4)),
                "chi2": float(m.group(5)),
                "unknown": float(m.group(6)),
                "chirp_factor": float(m.group(7)),
            })

        # [C++debug] lines
        if line.startswith("[C++debug]"):
            for kv in re.findall(r"(\w+)=([\d.eE+-]+)", line):
                result["debug"][kv[0]] = float(kv[1])
        if line.startswith("[C++debug2]"):
            for kv in re.findall(r"(\w+)=([\d.eE+-]+)", line):
                result["debug"][kv[0]] = float(kv[1])
        if line.startswith("[C++debug3]"):
            for kv in re.findall(r"(\w+)=([\d.eE+-]+)", line):
                result["debug"][kv[0]] = float(kv[1])
        if line.startswith("[C++debug4]"):
            # Per-IFO: ifo=N enrg=X sSNR=X ...
            ifo_match = re.match(r"\[C\+\+debug4\]\s+ifo=(\d+)\s+(.*)", line)
            if ifo_match:
                ifo_idx = int(ifo_match.group(1))
                for kv in re.findall(r"(\w+)=([\d.eE+-]+)", ifo_match.group(2)):
                    result["debug"][f"ifo{ifo_idx}_{kv[0]}"] = float(kv[1])
        if line.startswith("[C++debug5]"):
            for kv in re.findall(r"(\w+)=([\d.eE+-]+)", line):
                result["debug"]["d5_" + kv[0]] = float(kv[1])

    return result


def parse_native_log(log_path: str) -> dict:
    """Extract [Py-*] debug lines from the native pipeline log."""
    result = {}
    if not os.path.isfile(log_path):
        return result

    with open(log_path) as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("[Py-setAMP]"):
            for kv in re.findall(r"(\w+)=([\d.eE+-]+)", line):
                result["setAMP_" + kv[0]] = float(kv[1])
        if line.startswith("[Py-debug6]"):
            for kv in re.findall(r"(\w+)=([\d.eE+-]+)", line):
                result["d6_" + kv[0]] = float(kv[1])
        if line.startswith("[Py-mchirp]"):
            for kv in re.findall(r"(\w+)=([\d.eE+-]+)", line):
                result["mchirp_" + kv[0]] = float(kv[1])
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 5. HELPER: compute relative difference
# ═══════════════════════════════════════════════════════════════════════════

def reldiff(a: float, b: float) -> str:
    """Return relative difference if both nonzero, else absolute."""
    if a == 0 and b == 0:
        return "0"
    denom = max(abs(a), abs(b))
    if denom == 0:
        return f"abs={abs(a - b):.6g}"
    rd = abs(a - b) / denom
    return f"{rd:.6e} ({(a - b) / denom:+.4%})"


def compare_row(label: str, native_val, cwb_val, unit: str = ""):
    """Print a single comparison row."""
    if isinstance(native_val, float) and isinstance(cwb_val, float):
        diff_str = reldiff(native_val, cwb_val)
    else:
        diff_str = "N/A"
    unit_str = f" {unit}" if unit else ""
    print(f"  {label:<30s}  native={native_val:<20s}{unit_str}  "
          f"cwb={cwb_val:<20s}{unit_str}  diff={diff_str}")


def fmtf(v, prec=6):
    """Format a float to string with given precision."""
    if isinstance(v, float):
        return f"{v:.{prec}f}"
    return str(v)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("COMPARISON: pycWB Native vs cWB C++ (Standalone)")
    print("=" * 80)

    # ------------------------------------------------------------------
    # A. Native catalog (Parquet)
    # ------------------------------------------------------------------
    print("\n" + "─" * 80)
    print("A. NATIVE OUTPUT: Parquet catalog")
    print("─" * 80)
    if not os.path.isfile(CATALOG_PARQUET):
        print(f"  [ERROR] Catalog not found: {CATALOG_PARQUET}")
        sys.exit(1)

    df = read_catalog(CATALOG_PARQUET)
    print(f"  Events in catalog: {len(df)}")
    for idx, row in df.iterrows():
        print(f"\n  --- Event {idx} (id={row['id']}) ---")
        print(f"  cluster_id={row['cluster_id']}  n_pixels_total={row['n_pixels_total']}  "
              f"n_pixels_core={row['n_pixels_core']}")
        print(f"  rho={row['rho']:.6f}  rho_alt={row['rho_alt']:.6f}")
        print(f"  net_cc={row['net_cc']:.6f}  sky_cc={row['sky_cc']:.6f}  "
              f"subnet_cc={row['subnet_cc']:.6f}  subnet_cc2={row['subnet_cc2']:.6f}")
        print(f"  likelihood={row['likelihood']:.6f}  coherent_energy={row['coherent_energy']:.6f}  "
              f"ECOR={row['coherent_energy_norm']:.6f}")
        print(f"  net_energy_disb={row['net_energy_disb']:.6f}  net_null={row['net_null']:.6f}  "
              f"net_energy={row['net_energy']:.6f}")
        print(f"  like_sky={row['like_sky']:.6f}  energy_sky={row['energy_sky']:.6f}")
        print(f"  phi={row['phi']:.6f}  theta={row['theta']:.6f}  "
              f"ra={row['ra']:.6f}  dec={row['dec']:.6f}")
        print(f"  gps_time={row['gps_time']:.6f}")
        print(f"  mchirp={row['mchirp']:.6f}  chirp_ellip={row['chirp_ellip']:.6f}")
        print(f"  gnet={row['network_sensitivity']:.6f}  anet={row['network_alignment_factor']:.6f}")
        print(f"  strain={row['strain']:.6e}")
        print(f"  penalty={row['penalty']:.6f}  q_veto={row['q_veto']:.6f}")
        # Per-IFO
        for ifo in row["ifo_list"]:
            print(f"    {ifo}: time={row.get(f'time_{ifo}', 'N/A'):.6f}  "
                  f"freq={row.get(f'central_freq_{ifo}', 'N/A'):.4f}  "
                  f"hrss={row.get(f'hrss_{ifo}', 'N/A'):.6e}  "
                  f"data_E={row.get(f'data_energy_{ifo}', 'N/A'):.4f}  "
                  f"sig_E={row.get(f'signal_energy_{ifo}', 'N/A'):.4f}  "
                  f"cross_E={row.get(f'cross_energy_{ifo}', 'N/A'):.4f}  "
                  f"null_E={row.get(f'null_energy_{ifo}', 'N/A'):.6f}  "
                  f"fp={row.get(f'fp_{ifo}', 'N/A'):.6f}  "
                  f"fx={row.get(f'fx_{ifo}', 'N/A'):.6f}")

    # ------------------------------------------------------------------
    # B. Native trigger JSONs
    # ------------------------------------------------------------------
    print("\n" + "─" * 80)
    print("B. NATIVE OUTPUT: Trigger JSONs")
    print("─" * 80)
    triggers = read_trigger_jsons(TRIGGER_DIR)
    print(f"  Triggers found: {len(triggers)}")
    for t in triggers:
        meta = t["meta"]
        print(f"\n  --- {t['name']} (cluster_id={t['cluster_id']}, n_pixels={t['n_pixels']}) ---")
        print(f"  net_rho={meta.get('net_rho', 'N/A'):.6f}  "
              f"net_rho2={meta.get('net_rho2', 'N/A'):.6f}")
        print(f"  net_cc={meta.get('net_cc', 'N/A'):.6f}  "
              f"sky_cc={meta.get('sky_cc', 'N/A'):.6f}  "
              f"sub_net={meta.get('sub_net', 'N/A'):.6f}")
        print(f"  like_net={meta.get('like_net', 'N/A'):.6f}  "
              f"net_ecor={meta.get('net_ecor', 'N/A'):.6f}  "
              f"norm_cor={meta.get('norm_cor', 'N/A'):.6f}")
        print(f"  energy={meta.get('energy', 'N/A'):.6f}  "
              f"g_noise (net_null)={meta.get('g_noise', 'N/A'):.6f}")
        print(f"  net_ed={meta.get('net_ed', 'N/A'):.6f}")
        print(f"  phi={meta.get('phi', 'N/A'):.6f}  theta={meta.get('theta', 'N/A'):.6f}")
        print(f"  g_net={meta.get('g_net', 'N/A'):.6f}  a_net={meta.get('a_net', 'N/A'):.6f}")

    # ------------------------------------------------------------------
    # C. CWB ROOT supercluster data
    # ------------------------------------------------------------------
    print("\n" + "─" * 80)
    print("C. CWB OUTPUT: ROOT supercluster pixel data")
    print("─" * 80)
    if not os.path.isfile(CWB_ROOT_FILE):
        print(f"  [WARN] ROOT file not found: {CWB_ROOT_FILE}")
        cwb_clusters = {}
    else:
        cwb_clusters = read_cwb_supercluster(CWB_ROOT_FILE)
        total_pix = sum(c["n_pixels"] for c in cwb_clusters.values())
        print(f"  Total pixels: {total_pix}")
        print(f"  Clusters: {len(cwb_clusters)}")
        for ci, info in sorted(cwb_clusters.items()):
            print(f"\n  --- CWB Cluster {ci} ---")
            print(f"    n_pixels: {info['n_pixels']}")
            print(f"    rates: {info['rates']}")
            print(f"    ctime: {info['ctime']:.6f}")
            print(f"    cfreq: {info['cfreq']:.6f}")

    # ------------------------------------------------------------------
    # D. CWB log (mchirp_2g + debug)
    # ------------------------------------------------------------------
    print("\n" + "─" * 80)
    print("D. CWB OUTPUT: Log-extracted event data")
    print("─" * 80)
    cwb_log = parse_cwb_log(CWB_LOG)
    print(f"  mchirp_2g entries: {len(cwb_log['mchirp_2g'])}")
    for entry in cwb_log["mchirp_2g"]:
        print(f"    cluster_id={entry['cluster_id']}  mchirp={entry['mchirp']:.4f}  "
              f"chirp_ellip={entry['chirp_ellip']:.4f}  chirp_factor={entry['chirp_factor']:.4f}")
    if cwb_log["debug"]:
        print("  C++ debug values (from hybrid log):")
        for k, v in sorted(cwb_log["debug"].items()):
            print(f"    {k} = {v}")

    # Also parse hybrid log for C++ debug data
    hybrid_log = parse_cwb_log(HYBRID_LOG)
    if hybrid_log["debug"]:
        print("\n  Hybrid-log C++ debug values:")
        for k, v in sorted(hybrid_log["debug"].items()):
            print(f"    {k} = {v}")

    # ------------------------------------------------------------------
    # E. Native log debug data
    # ------------------------------------------------------------------
    native_debug = parse_native_log(NATIVE_LOG)
    if native_debug:
        print("\n  Native-log Python debug values:")
        for k, v in sorted(native_debug.items()):
            print(f"    {k} = {v}")

    # ==================================================================
    # COMPARISON TABLES
    # ==================================================================
    print("\n" + "=" * 80)
    print("COMPARISON: Event-Level Fields")
    print("=" * 80)

    # Match native events to CWB clusters by approximate time
    # Native parquet event times are absolute GPS; CWB cluster times are
    # relative to segment start. We use cluster_id matching.

    # ---------- Cluster-level comparison (pixel counts) ----------
    print("\n" + "─" * 80)
    print("TABLE 1: Cluster Pixel Counts (CWB supercluster vs Native catalog)")
    print("─" * 80)
    print(f"  {'Cluster':<10s}  {'CWB pixels':>12s}  {'Native total':>14s}  {'Native core':>13s}  "
          f"{'CWB ctime':>12s}  {'Native ctime':>14s}")
    print("  " + "─" * 80)

    # The CWB clusters are numbered 1,2; native catalog has cluster_id field
    for _, row in df.iterrows():
        cid = row["cluster_id"]
        cwb_info = cwb_clusters.get(cid, {})
        cwb_n = cwb_info.get("n_pixels", "N/A")
        cwb_ct = cwb_info.get("ctime", 0)
        # Native relative time: gps_time - gps_start
        # gps_start ≈ segment_start
        nat_ctime = row["gps_time"] - row.get("segment_start_L1", row["gps_time"])
        print(f"  {cid:<10d}  {str(cwb_n):>12s}  {row['n_pixels_total']:>14d}  "
              f"{row['n_pixels_core']:>13d}  {cwb_ct:>12.3f}  {nat_ctime:>14.3f}")

    # ---------- Likelihood-level comparison ----------
    # Compare hybrid C++ debug values with native debug values
    # Both come from the same cluster (the injection signal event)
    hybrid_dbg = hybrid_log.get("debug", {})
    if hybrid_dbg and native_debug:
        print("\n" + "─" * 80)
        print("TABLE 2: Likelihood Debug (C++ hybrid vs Python native)")
        print("         (Same input data, different likelihood implementations)")
        print("─" * 80)
        print(f"  {'Field':<30s}  {'C++ (hybrid)':>16s}  {'Python (native)':>16s}  {'Rel Diff':>16s}")
        print("  " + "─" * 80)

        comparisons = [
            ("rho (raw)",           hybrid_dbg.get("d5_rho"),      native_debug.get("d6_rho")),
            ("rho (reduced)",       None,                           native_debug.get("d6_rho_reduced")),
            ("cc",                  hybrid_dbg.get("d5_cc"),       native_debug.get("d6_cc_sky")),
            ("Ec (coherent energy)",hybrid_dbg.get("d5_Ec"),       native_debug.get("d6_Ec")),
            ("Rc (correlation)",    hybrid_dbg.get("d5_Rc"),       native_debug.get("d6_Rc")),
            ("Np (noise penalty)",  hybrid_dbg.get("d5_Np"),       native_debug.get("d6_Np")),
            ("Nw",                  hybrid_dbg.get("Nw"),          None),
            ("Ew (total energy)",   hybrid_dbg.get("Ew"),          None),
            ("Lw (likelihood)",     hybrid_dbg.get("Lw"),          None),
            ("netED",               hybrid_dbg.get("netED"),       None),
            ("Gn (g_net)",          hybrid_dbg.get("Gn"),          None),
            ("N (noise)",           hybrid_dbg.get("N"),           native_debug.get("setAMP_N")),
            # Per-IFO
            ("IFO0 enrg",          hybrid_dbg.get("ifo0_enrg"),   None),
            ("IFO0 sSNR",          hybrid_dbg.get("ifo0_sSNR"),   None),
            ("IFO0 xSNR",          hybrid_dbg.get("ifo0_xSNR"),   None),
            ("IFO0 null",          hybrid_dbg.get("ifo0_null"),   None),
            ("IFO1 enrg",          hybrid_dbg.get("ifo1_enrg"),   None),
            ("IFO1 sSNR",          hybrid_dbg.get("ifo1_sSNR"),   None),
            ("IFO1 xSNR",          hybrid_dbg.get("ifo1_xSNR"),   None),
            ("IFO1 null",          hybrid_dbg.get("ifo1_null"),   None),
            # mchirp
            ("mchirp",             None,                           native_debug.get("mchirp_m0")),
            ("chirpEllip",         None,                           native_debug.get("mchirp_chirpEllip")),
            ("rho0 (final)",       None,                           native_debug.get("mchirp_rho0")),
            ("rho1 (chirp-reduced)",None,                          native_debug.get("mchirp_rho1")),
        ]

        for label, cpp_val, py_val in comparisons:
            cpp_str = f"{cpp_val:.6f}" if cpp_val is not None else "—"
            py_str = f"{py_val:.6f}" if py_val is not None else "—"
            if cpp_val is not None and py_val is not None:
                diff = reldiff(cpp_val, py_val)
            else:
                diff = "—"
            print(f"  {label:<30s}  {cpp_str:>16s}  {py_str:>16s}  {diff:>16s}")

    # ---------- Full event comparison: catalog vs trigger JSONs ----------
    print("\n" + "─" * 80)
    print("TABLE 3: Native Event Parameters (Catalog Parquet)")
    print("         Comparing the two detected events")
    print("─" * 80)

    if len(df) >= 2:
        # Sort by cluster_id for consistent ordering
        df_sorted = df.sort_values("cluster_id").reset_index(drop=True)

        fields = [
            ("cluster_id",              "cluster_id",               "int"),
            ("n_pixels_total",          "n_pixels_total",           "int"),
            ("n_pixels_core",           "n_pixels_core",            "int"),
            ("rho",                     "rho",                      "float"),
            ("rho_alt",                 "rho_alt",                  "float"),
            ("net_cc",                  "net_cc",                   "float"),
            ("sky_cc",                  "sky_cc",                   "float"),
            ("subnet_cc",              "subnet_cc",                "float"),
            ("subnet_cc2",             "subnet_cc2",               "float"),
            ("likelihood",              "likelihood",               "float"),
            ("coherent_energy",         "coherent_energy",          "float"),
            ("coherent_energy_norm",    "coherent_energy_norm",     "float"),
            ("net_energy_disb",         "net_energy_disb",          "float"),
            ("net_null",                "net_null",                 "float"),
            ("net_energy",              "net_energy",               "float"),
            ("like_sky",                "like_sky",                 "float"),
            ("energy_sky",              "energy_sky",               "float"),
            ("network_sensitivity",     "network_sensitivity",      "float"),
            ("network_alignment_factor","network_alignment_factor", "float"),
            ("phi",                     "phi",                      "float"),
            ("theta",                   "theta",                    "float"),
            ("ra",                      "ra",                       "float"),
            ("dec",                     "dec",                      "float"),
            ("penalty",                 "penalty",                  "float"),
            ("strain",                  "strain",                   "float_sci"),
            ("mchirp",                  "mchirp",                   "float"),
            ("chirp_ellip",             "chirp_ellip",              "float"),
            ("q_veto",                  "q_veto",                   "float"),
            ("q_factor",                "q_factor",                 "float"),
            ("gps_time",                "gps_time",                 "float"),
        ]

        print(f"  {'Field':<30s}", end="")
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            print(f"  {'Event ' + str(row['cluster_id']):>20s}", end="")
        print()
        print("  " + "─" * (30 + 22 * len(df_sorted)))

        for label, col, fmt in fields:
            print(f"  {label:<30s}", end="")
            for _, row in df_sorted.iterrows():
                val = row[col]
                if fmt == "int":
                    print(f"  {int(val):>20d}", end="")
                elif fmt == "float_sci":
                    print(f"  {val:>20.6e}", end="")
                else:
                    print(f"  {val:>20.6f}", end="")
            print()

    # ---------- Compare with CWB mchirp ----------
    if cwb_log["mchirp_2g"]:
        print("\n" + "─" * 80)
        print("TABLE 4: Chirp Mass Comparison (CWB standalone C++ vs Native)")
        print("─" * 80)
        print(f"  {'Cluster':<10s}  {'CWB mchirp':>12s}  {'CWB ellip':>10s}  "
              f"{'CWB factor':>11s}  {'Native mchirp':>14s}  {'Native ellip':>13s}")
        print("  " + "─" * 75)

        for cwb_entry in cwb_log["mchirp_2g"]:
            cid = cwb_entry["cluster_id"]
            # Find matching native event
            nat_rows = df[df["cluster_id"] == cid]
            if len(nat_rows) > 0:
                nat = nat_rows.iloc[0]
                nat_m = f"{nat['mchirp']:.4f}"
                nat_e = f"{nat['chirp_ellip']:.4f}"
            else:
                nat_m = "N/A"
                nat_e = "N/A"
            print(f"  {cid:<10d}  {cwb_entry['mchirp']:>12.4f}  {cwb_entry['chirp_ellip']:>10.4f}  "
                  f"{cwb_entry['chirp_factor']:>11.4f}  {nat_m:>14s}  {nat_e:>13s}")

    # ---------- Compare CWB pixel counts vs Native pixel counts ----------
    print("\n" + "─" * 80)
    print("TABLE 5: Pixel Count Comparison Summary")
    print("─" * 80)
    print(f"  {'Source':<25s}  {'Cluster 1 pix':>14s}  {'Cluster 2 pix':>14s}  {'Total pix':>12s}")
    print("  " + "─" * 70)

    # CWB standalone
    cwb_total = sum(c["n_pixels"] for c in cwb_clusters.values())
    cwb_c1 = cwb_clusters.get(1, {}).get("n_pixels", 0)
    cwb_c2 = cwb_clusters.get(2, {}).get("n_pixels", 0)
    print(f"  {'CWB standalone':<25s}  {cwb_c1:>14d}  {cwb_c2:>14d}  {cwb_total:>12d}")

    # From CWB log (mchirp_2g cluster sizes are in the log)
    # Parse "cluster-id|pixels:     1|414" from CWB log
    cwb_log_sizes = {}
    if os.path.isfile(CWB_LOG):
        with open(CWB_LOG) as f:
            for line in f:
                m = re.search(r"cluster-id\|pixels:\s+(\d+)\|(\d+)", line)
                if m:
                    cwb_log_sizes[int(m.group(1))] = int(m.group(2))
    if cwb_log_sizes:
        total_log = sum(cwb_log_sizes.values())
        print(f"  {'CWB log (likelihood)':<25s}  "
              f"{cwb_log_sizes.get(1, 0):>14d}  "
              f"{cwb_log_sizes.get(2, 0):>14d}  {total_log:>12d}")

    # Native catalog
    for _, row in df.sort_values("cluster_id").iterrows():
        pass  # just to get sorted
    df_s = df.sort_values("cluster_id")
    nat_c1 = int(df_s[df_s["cluster_id"] == 1]["n_pixels_total"].iloc[0]) if 1 in df_s["cluster_id"].values else 0
    nat_c2 = int(df_s[df_s["cluster_id"] == 2]["n_pixels_total"].iloc[0]) if 2 in df_s["cluster_id"].values else 0
    print(f"  {'Native (catalog total)':<25s}  {nat_c1:>14d}  {nat_c2:>14d}  {nat_c1 + nat_c2:>12d}")

    nat_core1 = int(df_s[df_s["cluster_id"] == 1]["n_pixels_core"].iloc[0]) if 1 in df_s["cluster_id"].values else 0
    nat_core2 = int(df_s[df_s["cluster_id"] == 2]["n_pixels_core"].iloc[0]) if 2 in df_s["cluster_id"].values else 0
    print(f"  {'Native (catalog core)':<25s}  {nat_core1:>14d}  {nat_core2:>14d}  "
          f"{nat_core1 + nat_core2:>12d}")

    # From native log
    nat_log_sizes = {}
    if os.path.isfile(NATIVE_LOG):
        with open(NATIVE_LOG) as f:
            for line in f:
                m = re.search(r"cluster-id\|pixels:\s+(\d+)\|(\d+)", line)
                if m:
                    nat_log_sizes[int(m.group(1))] = int(m.group(2))
    if nat_log_sizes:
        total_nat_log = sum(nat_log_sizes.values())
        print(f"  {'Native log (likelihood)':<25s}  "
              f"{nat_log_sizes.get(1, 0):>14d}  "
              f"{nat_log_sizes.get(2, 0):>14d}  {total_nat_log:>12d}")

    # ---------- Summary ----------
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("  Data sources:")
    print(f"    Native catalog:  {CATALOG_PARQUET}  ({len(df)} events)")
    print(f"    CWB ROOT:        {CWB_ROOT_FILE}  ({len(cwb_clusters)} clusters, {cwb_total} pixels)")
    print(f"    CWB log:         {CWB_LOG}  ({len(cwb_log['mchirp_2g'])} mchirp_2g entries)")
    print(f"    Hybrid log:      {HYBRID_LOG}")
    print(f"    Native log:      {NATIVE_LOG}")
    print()
    print("  Notes:")
    print("  - The CWB ROOT file contains SUPERCLUSTER-level data (pixel lists),")
    print("    not final event-level (waveburst) parameters.")
    print("  - The wave_*.root.tmp event file appears incomplete — CWB may not have")
    print("    finished writing event data before the job completed.")
    print("  - Event-level CWB results are extracted from log debug prints instead.")
    print("  - The hybrid pipeline (pycWB + C++ likelihood) provides the closest")
    print("    C++ event-level data for comparison.")
    print()

    # Key metrics from hybrid vs native (if both available)
    if hybrid_dbg and native_debug:
        print("  KEY METRICS (C++ hybrid vs Python native, same-data comparison):")
        pairs = [
            ("rho",   hybrid_dbg.get("d5_rho"),  native_debug.get("d6_rho")),
            ("cc",    hybrid_dbg.get("d5_cc"),   native_debug.get("d6_cc_sky")),
            ("Ec",    hybrid_dbg.get("d5_Ec"),   native_debug.get("d6_Ec")),
            ("Rc",    hybrid_dbg.get("d5_Rc"),   native_debug.get("d6_Rc")),
        ]
        for name, cpp_v, py_v in pairs:
            if cpp_v is not None and py_v is not None:
                rd = abs(cpp_v - py_v) / max(abs(cpp_v), abs(py_v), 1e-30)
                status = "OK" if rd < 0.01 else ("WARN" if rd < 0.05 else "MISMATCH")
                print(f"    {name:<6s}: C++={cpp_v:.6f}  PY={py_v:.6f}  "
                      f"relDiff={rd:.2e}  [{status}]")
        print()


if __name__ == "__main__":
    main()
