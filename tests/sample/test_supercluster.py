import copy
import os
import sys
from pathlib import Path

import ROOT
import numpy as np
from wdm_wavelet.wdm import WDM as WDMWavelet

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pycwb.config import Config
from pycwb.modules.coherence import coherence as coherence_cwb
from pycwb.modules.cwb_coherence.lag_plan import build_lag_plan_from_config
from pycwb.modules.data_conditioning import data_conditioning as data_conditioning_cwb
from pycwb.modules.multi_resolution_wdm import create_wdm_for_level
from pycwb.modules.read_data import generate_injection, generate_noise_for_job_seg
from pycwb.modules.read_data.data_check import check_and_resample
from pycwb.modules.job_segment import create_job_segment_from_config
from pycwb.modules.sparse_series import sparse_table_from_fragment_clusters
from pycwb.modules.super_cluster.sub_net_cut import sub_net_cut
from pycwb.modules.super_cluster.super_cluster import supercluster as native_supercluster
from pycwb.modules.super_cluster.super_cluster import defragment as native_defragment
from pycwb.modules.xtalk.monster import load_catalog
from pycwb.modules.cwb_conversions import (
    convert_fragment_clusters_to_netcluster,
    convert_netcluster_to_fragment_clusters,
    convert_sparse_series_to_sseries,
)
from pycwb.types.detector import compute_sky_delay_and_patterns, calculate_e2or_from_acore
from pycwb.types.network import Network


SAMPLE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SAMPLE_DIR / "user_parameters_injection.yaml"


def _pixel_signature(pixel):
    indices = tuple(int(d.index) for d in pixel.data)
    return (int(pixel.time), float(pixel.frequency), int(pixel.rate), int(pixel.layers), indices)


def _cluster_signatures(clusters):
    signatures = []
    for cluster in clusters:
        pix_sigs = sorted(_pixel_signature(p) for p in cluster.pixels)
        signatures.append(tuple(pix_sigs))
    signatures.sort(key=lambda item: (len(item), item[0] if item else ()))
    return signatures


def _prepare_fragment_clusters_from_cwb_coherence():
    config = Config()
    # Change to sample dir so relative paths in YAML work
    old_cwd = Path.cwd()
    try:
        import os
        os.chdir(SAMPLE_DIR)
        config.load_from_yaml(str(CONFIG_PATH))
        config.nproc = 1
        job_segments = create_job_segment_from_config(config)
    finally:
        os.chdir(old_cwd)
    data = generate_noise_for_job_seg(job_segments[0], config.inRate, f_low=config.fLow)
    data = generate_injection(config, job_segments[0], data)
    data = [check_and_resample(data[i], config, i) for i in range(len(job_segments[0].ifos))]

    strains_cwb, nRMS_cwb = data_conditioning_cwb(config, data)

    network = Network(config, strains_cwb, nRMS_cwb, silent=True)
    fragment_clusters = coherence_cwb(config, strains_cwb, nRMS_cwb, net=network)

    return config, strains_cwb, nRMS_cwb, fragment_clusters


def _run_cwb_supercluster_with_stage_counts(config, strains, nRMS_list, fragment_clusters):
    tf_maps = strains
    network = Network(config, tf_maps, nRMS_list, silent=True)
    sparse_table_list = sparse_table_from_fragment_clusters(config, tf_maps, fragment_clusters)

    skyres = config.MIN_SKYRES_HEALPIX if config.healpix > config.MIN_SKYRES_HEALPIX else 0
    if skyres > 0:
        network.update_sky_map(config, skyres)
        network.net.setAntenna()
        network.net.setDelay(config.refIFO)
        network.update_sky_mask(config, skyres)

    hot = [network.get_ifo(n).getHoT() for n in range(config.nIFO)]

    for level in config.WDM_level:
        wdm = create_wdm_for_level(config, level)
        wdm.set_td_filter(config.TDSize, 1)
        network.add_wavelet(wdm)

    for n in range(config.nIFO):
        det = network.get_ifo(n)
        det.sclear()
        for sparse_table in sparse_table_list:
            det.vSS.push_back(convert_sparse_series_to_sseries(sparse_table[n]))

    stage_counts = []
    final_clusters = []
    cwb_super_e2or = float(network.net.e2or)
    cwb_pattern = int(network.pattern)

    for lag in range(int(network.nLag)):
        stage = {"lag": lag}

        merged = copy.deepcopy(fragment_clusters[0][lag])
        if len(fragment_clusters) > 1:
            for fragment_cluster in fragment_clusters[1:]:
                merged.clusters += fragment_cluster[lag].clusters

        net_cluster = convert_fragment_clusters_to_netcluster(merged)
        stage["merged_clusters"] = int(net_cluster.esize(0))

        net_cluster.supercluster("L", network.net.e2or, config.TFgap, False)
        cwb_super_fc = convert_netcluster_to_fragment_clusters(net_cluster)
        stage["super_clusters"] = int(len(cwb_super_fc.clusters))
        stage["super_sizes"] = [len(c.pixels) for c in cwb_super_fc.clusters]
        stage["super_signatures"] = _cluster_signatures(cwb_super_fc.clusters)

        pwc = network.get_cluster(lag)
        pwc.cpf(net_cluster, False)

        if network.pattern == 0 or config.subnet > 0 or config.subcut > 0 or config.subnorm > 0 or config.subrho >= 0:
            if config.subacor > 0:
                network.net.acor = config.subacor
            if config.subrho > 0:
                network.net.netRHO = config.subrho

            network.set_delay_index(hot[0].rate())
            pwc.setcore(False)

            while True:
                loaded = pwc.loadTDampSSE(network.net, "a", config.BATCH, config.LOUD)
                network.sub_net_cut(lag, config.subnet, config.subcut, config.subnorm)
                if loaded < config.BATCH:
                    break

            if config.subacor > 0:
                network.net.acor = config.Acore
            if config.subrho > 0:
                network.net.netRHO = config.netRHO

        post_subnet = convert_netcluster_to_fragment_clusters(pwc)
        stage["post_subnet_clusters"] = int(post_subnet.event_count())

        if network.pattern == 0:
            pwc.defragment(config.Tgap, config.Fgap)

        post_defrag = convert_netcluster_to_fragment_clusters(pwc)
        stage["post_defrag_clusters"] = int(post_defrag.event_count())

        post_defrag.remove_rejected()
        stage["final_clusters"] = int(post_defrag.event_count())

        final_clusters.append(post_defrag)
        stage_counts.append(stage)

        pwc.clean(1)
        pwc.clear()

    network.restore_skymap(config, skyres)
    return final_clusters, stage_counts, cwb_super_e2or, cwb_pattern


def _run_native_supercluster_with_stage_counts(config, fragment_clusters, strains, xtalk_coeff, xtalk_lookup_table, layers, super_e2or_override=None, pattern_override=None):
    tf_maps = strains
    def _extract_timeseries_data(tf_series):
        values = np.asarray(tf_series.data, dtype=np.float64)
        sample_rate = float(tf_series.sample_rate)
        start = float(tf_series.start)
        return values, sample_rate, start

    def _expected_td_vec_len(td_size):
        return 4 * int(td_size) + 2

    def _normalize_wdm_layers(layer_tag):
        layer_tag = int(layer_tag)
        if layer_tag <= 1:
            return 1
        # cWB pixel convention stores layers as (wdm_M + 1)
        candidate = layer_tag - 1
        return candidate if candidate % 2 == 0 else layer_tag

    def _resolve_wdm_context(layer_tag, context_map):
        layer_tag = int(layer_tag)
        context = context_map.get(layer_tag)
        if context is not None:
            return context
        context = context_map.get(layer_tag - 1)
        if context is not None:
            return context
        raise KeyError(f"Missing WDM context for layer tag {layer_tag}")

    def _apply_subnet_cut(superclusters, n_loudest_local, ml_local, FP_local, FX_local,
                          acor_local, e2or_local, n_ifo_local, n_sky_local,
                          subnet_local, subcut_local, subnorm_local, subrho_local,
                          xtalk_coeff_local, xtalk_lookup_table_local, layers_local):
        for c in superclusters:
            c.pixels.sort(key=lambda x: x.likelihood, reverse=True)
            results = sub_net_cut(
                c.pixels[:n_loudest_local], ml_local, FP_local, FX_local,
                acor_local, e2or_local, n_ifo_local, n_sky_local,
                subnet_local, subcut_local, subnorm_local, subrho_local,
                xtalk_coeff_local, xtalk_lookup_table_local, layers_local,
            )
            c.cluster_status = -1 if (
                results["subnet_passed"] and results["subrho_passed"] and results["subthr_passed"]
            ) else 1
        return [c for c in superclusters if c.cluster_status <= 0]

    lag_plan = build_lag_plan_from_config(config, tf_maps)
    n_lag = int(lag_plan.n_lag)

    # Collect all unique layer values that appear in the fragments
    merged_by_lag = []
    for lag in range(n_lag):
        merged = copy.deepcopy(fragment_clusters[0][lag])
        if len(fragment_clusters) > 1:
            for fragment_cluster in fragment_clusters[1:]:
                merged.clusters += fragment_cluster[lag].clusters
        merged_by_lag.append(merged)
    
    unique_wdm_layers = set()
    for fragment_cluster in merged_by_lag:
        for cluster in fragment_cluster.clusters:
            for pixel in cluster.pixels:
                unique_wdm_layers.add(int(_normalize_wdm_layers(pixel.layers)))

    # Build WDM contexts for all required layers
    wdm_context_by_layers = {}
    for layer_count in sorted(unique_wdm_layers):
        wdm = WDMWavelet(
            M=layer_count,
            K=layer_count,
            beta_order=config.WDM_beta_order,
            precision=config.WDM_precision,
        )
        wdm.set_td_filter(int(config.TDSize), 1)
        detector_tf_maps = []
        for n in range(config.nIFO):
            ts_data, sample_rate, t0 = _extract_timeseries_data(tf_maps[n])
            detector_tf_maps.append(wdm.t2w(ts_data, sample_rate=sample_rate, t0=t0, MM=-1))
        context = {"wdm": wdm, "tf_maps": detector_tf_maps}
        wdm_context_by_layers[int(layer_count)] = context
        wdm_context_by_layers[int(layer_count) + 1] = context

    td_vec_default = np.zeros(_expected_td_vec_len(config.TDSize), dtype=np.float32)
    for fragment_cluster in merged_by_lag:
        for cluster in fragment_cluster.clusters:
            for pixel in cluster.pixels:
                context = _resolve_wdm_context(pixel.layers, wdm_context_by_layers)
                wdm = context["wdm"]
                detector_tf_maps = context["tf_maps"]
                pixel_td_amp = []
                for n in range(config.nIFO):
                    try:
                        td_vec = wdm.get_td_vec(
                            detector_tf_maps[n],
                            pixel_index=int(pixel.data[n].index),
                            K=int(config.TDSize),
                            mode="a",
                        )
                        pixel_td_amp.append(np.asarray(td_vec, dtype=np.float32))
                    except Exception:
                        pixel_td_amp.append(td_vec_default.copy())
                pixel.td_amp = pixel_td_amp

    super_acor = config.Acore
    super_e2or = float(super_e2or_override) if super_e2or_override is not None else calculate_e2or_from_acore(super_acor, config.nIFO)
    subnet_acor = config.subacor if config.subacor > 0 else config.Acore
    subnet_e2or = calculate_e2or_from_acore(subnet_acor, config.nIFO)
    ml, FP, FX = compute_sky_delay_and_patterns(
        ifos=config.ifo,
        ref_ifo=config.refIFO,
        sample_rate=config.rateANA,
        td_size=int(config.TDSize),
        gps_time=float(tf_maps[0].start),
        healpix_order=int(config.healpix) if hasattr(config, "healpix") else None,
        n_sky=None,
    )

    n_sky = int(ml.shape[1])
    n_ifo = config.nIFO
    n_loudest = config.LOUD
    subrho = config.subrho if config.subrho > 0 else config.netRHO
    pattern = int(pattern_override) if pattern_override is not None else int(getattr(config, "pattern", 0))

    stage_counts = []
    final_clusters = []

    for lag, fragment_cluster in enumerate(merged_by_lag):
        stage = {"lag": lag}
        clusters = fragment_cluster.clusters
        stage["merged_clusters"] = int(len(clusters))

        superclusters = native_supercluster(clusters, "L", config.TFgap, super_e2or, n_ifo)
        stage["super_clusters"] = int(len(superclusters))
        stage["super_sizes"] = [len(c.pixels) for c in superclusters]
        stage["super_signatures"] = _cluster_signatures(superclusters)

        accepted = [sc for sc in superclusters if sc.cluster_status <= 0]
        if len(accepted) == 0:
            stage["post_subnet_clusters"] = 0
            stage["post_defrag_clusters"] = 0
            stage["final_clusters"] = 0
            stage_counts.append(stage)
            final_clusters.append(fragment_cluster)
            continue

        selected = _apply_subnet_cut(
            accepted, n_loudest, ml, FP, FX,
            subnet_acor, subnet_e2or, n_ifo, n_sky,
            config.subnet, config.subcut, config.subnorm, subrho,
            xtalk_coeff, xtalk_lookup_table, layers,
        )
        stage["post_subnet_clusters"] = int(len(selected))

        if pattern == 0:
            defragged = native_defragment(selected, config.Tgap, config.Fgap, n_ifo)
            stage["post_defrag_clusters"] = int(len(defragged))
            final = defragged
        else:
            stage["post_defrag_clusters"] = int(len(selected))
            final = selected

        for c in final:
            for p in c.pixels:
                p.core = 1
                p.td_amp = None

        fragment_cluster.clusters = final
        stage["final_clusters"] = int(len(final))
        stage_counts.append(stage)
        final_clusters.append(fragment_cluster)

    return final_clusters, stage_counts


def _format_stage_table(cwb_stage_counts, native_stage_counts):
    lines = [
        "lag | merged(cwb/native) | super(cwb/native) | subnet(cwb/native) | defrag(cwb/native) | final(cwb/native)",
        "----|---------------------|-------------------|--------------------|--------------------|------------------",
    ]

    for cwb_row, native_row in zip(cwb_stage_counts, native_stage_counts):
        lines.append(
            f"{cwb_row['lag']:3d} | "
            f"{cwb_row['merged_clusters']:4d}/{native_row['merged_clusters']:4d} | "
            f"{cwb_row['super_clusters']:4d}/{native_row['super_clusters']:4d} | "
            f"{cwb_row['post_subnet_clusters']:4d}/{native_row['post_subnet_clusters']:4d} | "
            f"{cwb_row['post_defrag_clusters']:4d}/{native_row['post_defrag_clusters']:4d} | "
            f"{cwb_row['final_clusters']:4d}/{native_row['final_clusters']:4d}"
        )

    return "\n".join(lines)


def test_supercluster_stage_counts_match_cwb():
    config, strains_cwb, nRMS_cwb, fragment_clusters = _prepare_fragment_clusters_from_cwb_coherence()

    cwb_fragments, cwb_stage_counts, cwb_super_e2or, cwb_pattern = _run_cwb_supercluster_with_stage_counts(
        config=config,
        strains=strains_cwb,
        nRMS_list=nRMS_cwb,
        fragment_clusters=copy.deepcopy(fragment_clusters),
    )

    xtalk_coeff, xtalk_lookup_table, layers, _ = load_catalog(config.MRAcatalog)
    native_fragments, native_stage_counts = _run_native_supercluster_with_stage_counts(
        config=config,
        fragment_clusters=copy.deepcopy(fragment_clusters),
        strains=strains_cwb,
        xtalk_coeff=xtalk_coeff,
        xtalk_lookup_table=xtalk_lookup_table,
        layers=layers,
        super_e2or_override=cwb_super_e2or,
        pattern_override=cwb_pattern,
    )

    assert len(cwb_stage_counts) == len(native_stage_counts), "Lag count mismatch between cWB and native stages"

    mismatches = []
    compare_keys = [
        "merged_clusters",
        "super_clusters",
        "post_subnet_clusters",
        "post_defrag_clusters",
        "final_clusters",
    ]

    for cwb_row, native_row in zip(cwb_stage_counts, native_stage_counts):
        for key in compare_keys:
            if int(cwb_row[key]) != int(native_row[key]):
                mismatches.append((cwb_row["lag"], key, int(cwb_row[key]), int(native_row[key])))
    stage_table = _format_stage_table(cwb_stage_counts, native_stage_counts)
    print("Stage comparison between cWB and native supercluster:")
    print(stage_table)
    if mismatches:
        mismatch_lines = [f"lag={lag}, stage={stage}, cwb={cwb_val}, native={native_val}" for lag, stage, cwb_val, native_val in mismatches]
        signature_lines = []
        for cwb_row, native_row in zip(cwb_stage_counts, native_stage_counts):
            if int(cwb_row["super_clusters"]) != int(native_row["super_clusters"]):
                signature_lines.append(
                    f"lag={cwb_row['lag']} super_sizes cwb={cwb_row.get('super_sizes', [])} native={native_row.get('super_sizes', [])}"
                )
        raise AssertionError(
            "Supercluster stage-count mismatch detected:\n"
            + "\n".join(mismatch_lines)
            + ("\n\nSupercluster partition debug:\n" + "\n".join(signature_lines) if signature_lines else "")
            + "\n\nStage table:\n"
            + stage_table
        )

    assert sum(fc.event_count() for fc in cwb_fragments) == sum(fc.event_count() for fc in native_fragments)

if __name__ == "__main__":
    test_supercluster_stage_counts_match_cwb()
