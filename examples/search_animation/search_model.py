"""Didactic coherent-search model used by ``render_search_animation.py``.

Every quantity drawn in the animation is computed here from simulated data.
The steps follow the order of the pycWB native search
(``pycwb/workflow/subflow/process_job_segment_native.py``), each reduced to a
few lines of NumPy so it can be read alongside the animation:

1. project ``h+`` into every detector with PyCBC (antenna pattern + delay),
2. colour white noise with an aLIGO-like ASD and add the projected signal,
3. whiten: per-layer WDM noise RMS ``sqrt(0.7191 * median(00^2 + 90^2))``,
4. WDM transforms at several resolutions,
5. per-detector energy maximised over +-light-travel-time shifts, network
   energy threshold from the black-pixel probability ``bpp``, neighbour
   support, 8-neighbour clustering,
6. link clusters from different resolutions (TF-gap metric), defragment and
   apply a sub-network energy cut,
7. sky loop over HEALPix: sky delays select time-delayed pixel amplitudes,
   antenna patterns are rotated to the dominant polarisation frame (DPF) and
   the data are projected on the regulated network response; the sky
   statistic is ``Lo * cc``,
8. reconstruct the whitened waveform at the best sky location,
9. repeat 4-7 on circular time slides of the other detectors to build a
   background distribution of the coherent SNR ``rho``.

The likelihood is a compact version of the cWB 2G statistic: it keeps the
DPF, the regulator on the weak polarisation, coherent/null energies and
``rho = sqrt(Ec * cc / (nIFO - 1))``, but not the xtalk/MRA corrections, the
packet (pattern) logic or the chirp/Q-veto post-processing of
``pycwb/modules/likelihoodWP``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

NRMS_MEDIAN_FACTOR = 0.7191


@dataclass
class SearchConfig:
    ifos: tuple[str, ...] = ("H1", "L1")
    gps_time: float = 1126259462.0
    source_ra: float = 1.4
    source_dec: float = 0.3
    source_polarization: float = 0.0
    noise_sigma: float = 0.09
    noise_seed: int = 7
    pad_seconds: float = 3.0
    edge_seconds: float = 0.5
    levels: tuple[int, ...] = (16, 32, 64, 128)
    display_level: int = 64
    whiten_level: int = 256
    wdm_beta_order: int = 6
    wdm_precision: int = 10
    f_low: float = 32.0
    f_high: float = 1024.0
    bpp: float = 0.001
    max_energy_step: int = 4
    td_upsample: int = 4
    tf_gap: float = 6.0
    t_gap: float = 3.0
    f_gap: float = 130.0
    min_pixels: int = 3
    subnet_cut: float = 0.1
    acore: float = float(np.sqrt(2.0))
    delta: float = 0.5
    nside: int = 16
    net_cc_cut: float = 0.5
    net_rho_cut: float = 4.0
    lag_step: float = 0.25
    n_lags: int = 31


@dataclass
class ResolutionMaps:
    """Products of one WDM resolution for one lag (full-length TF arrays)."""

    level: int
    dt: float
    df: float
    coeff: np.ndarray  # (n_det, n_freq, n_time) zero-delay coefficients
    energy_max: np.ndarray  # (n_det, n_freq, n_time) max over shifts
    net_energy: np.ndarray  # (n_freq, n_time)
    threshold: float
    core: np.ndarray
    selected: np.ndarray
    labels: np.ndarray

    @property
    def n_freq(self) -> int:
        return self.coeff.shape[1]

    @property
    def n_time(self) -> int:
        return self.coeff.shape[2]


@dataclass
class Fragment:
    res: int
    label: int
    f_idx: np.ndarray
    t_idx: np.ndarray


@dataclass
class Supercluster:
    fragments: list[int]
    res: np.ndarray
    f_idx: np.ndarray
    t_idx: np.ndarray
    time: np.ndarray
    freq: np.ndarray
    dt: np.ndarray
    df: np.ndarray
    det_excess: np.ndarray
    subnet: float
    passed: bool


@dataclass
class SkyScan:
    ra: np.ndarray
    dec: np.ndarray
    fplus: np.ndarray  # (n_det, n_sky)
    fcross: np.ndarray
    rel_delay: np.ndarray  # (n_det, n_sky) seconds, relative to ifos[0]
    shift: np.ndarray  # (n_det, n_sky) integer samples
    energy: np.ndarray
    likelihood: np.ndarray
    coherent: np.ndarray
    null: np.ndarray
    net_cc: np.ndarray
    statistic: np.ndarray
    best: int


@dataclass
class Trigger:
    cluster: int
    time: float
    f_low: float
    f_high: float
    ra: float
    dec: float
    norm: float
    energy: float
    likelihood: float
    coherent: float
    null: float
    net_cc: float
    rho: float
    rel_delay: np.ndarray
    passed: bool


@dataclass
class LagAnalysis:
    lag: float
    resolutions: list[ResolutionMaps]
    fragments: list[Fragment]
    links: list[tuple[int, int]]
    superclusters: list[Supercluster]
    triggers: list[Trigger]
    sky: dict[int, SkyScan]
    td_amplitudes: dict[int, np.ndarray]
    noise_mean: list[np.ndarray]


@dataclass
class SearchResult:
    config: SearchConfig
    sample_rate: float
    time: np.ndarray  # full series time axis (s from series start)
    window: tuple[float, float]  # display window in series time
    hplus: np.ndarray
    antenna: np.ndarray  # (n_det, 2) F+, Fx at the source
    arrival_delay: np.ndarray  # (n_det,) seconds from the geocentre
    clean: np.ndarray
    white_clean: np.ndarray
    raw: np.ndarray
    whitened: np.ndarray
    white_freq: np.ndarray
    white_rms: np.ndarray  # (n_det, n_layers)
    asd_at_layers: np.ndarray
    raw_display: np.ndarray  # (n_det, n_freq, n_time) raw energy, display level
    white_display: np.ndarray  # whitened energy, display level
    zero_lag: LagAnalysis
    event: Trigger
    reconstructed: np.ndarray  # (n_det, n_samples) whitened waveform
    recon_energy: np.ndarray  # (n_freq, n_time) display resolution
    null_energy: np.ndarray
    source_sky_index: int
    lags: np.ndarray
    background: list[list[Trigger]] = field(default_factory=list)

    def level_index(self, level: int) -> int:
        return list(self.config.levels).index(level)


# ---------------------------------------------------------------------------
# helpers


class WDMCache:
    def __init__(self, cfg: SearchConfig):
        self.cfg = cfg
        self._wdm: dict[int, object] = {}

    def __call__(self, level: int):
        if level not in self._wdm:
            from wdm_wavelet.wdm import WDM

            cfg = self.cfg
            try:
                self._wdm[level] = WDM(level, level, cfg.wdm_beta_order, cfg.wdm_precision, backend="jax")
            except TypeError:
                self._wdm[level] = WDM(level, level, cfg.wdm_beta_order, cfg.wdm_precision)
        return self._wdm[level]


def pixel_energy(coeff: np.ndarray) -> np.ndarray:
    """Pixel energy (00^2 + 90^2) / 2: unit mean for unit-variance white noise."""
    return 0.5 * (coeff.real ** 2 + coeff.imag ** 2)


def load_logo_waveform(path: Path) -> tuple[np.ndarray, float]:
    raw = np.loadtxt(path)
    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError(f"Expected two-column waveform file: {path}")
    time = raw[:, 0].astype(float)
    dt = float(np.median(np.diff(time)))
    if not np.allclose(np.diff(time), dt, rtol=1e-7, atol=1e-12):
        raise ValueError("Input waveform must have a uniform sample cadence")
    signal = raw[:, 1].astype(float)
    peak = np.max(np.abs(signal))
    if peak <= 0:
        raise ValueError("Input waveform is all zeros")
    return signal / peak, 1.0 / dt


def detectors(cfg: SearchConfig):
    from pycbc.detector import Detector

    return [Detector(ifo) for ifo in cfg.ifos]


def max_light_travel_time(cfg: SearchConfig) -> float:
    dets = detectors(cfg)
    return max(
        (a.light_travel_time_to_detector(b) for i, a in enumerate(dets) for b in dets[i + 1 :]),
        default=0.0,
    )


def reference_gps(cfg: SearchConfig, window_length: float) -> float:
    return cfg.gps_time + 0.5 * window_length


# ---------------------------------------------------------------------------
# 1. projection


def project_to_detectors(
    hplus: np.ndarray, sample_rate: float, cfg: SearchConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from pycbc.types import TimeSeries

    delta_t = 1.0 / sample_rate
    hp = TimeSeries(hplus, delta_t=delta_t, epoch=cfg.gps_time)
    hc = TimeSeries(np.zeros_like(hplus), delta_t=delta_t, epoch=cfg.gps_time)
    target = cfg.gps_time + np.arange(hplus.size) * delta_t
    ref_time = reference_gps(cfg, hplus.size * delta_t)

    projected, antenna, delays = [], [], []
    for det in detectors(cfg):
        strain = det.project_wave(
            hp, hc, cfg.source_ra, cfg.source_dec, cfg.source_polarization,
            method="constant", reference_time=ref_time,
        )
        sample_times = np.asarray(strain.sample_times, dtype=np.float64)
        values = np.asarray(strain.numpy(), dtype=np.float64)
        projected.append(np.interp(target, sample_times, values, left=0.0, right=0.0))
        antenna.append(det.antenna_pattern(cfg.source_ra, cfg.source_dec, cfg.source_polarization, ref_time))
        delays.append(det.time_delay_from_earth_center(cfg.source_ra, cfg.source_dec, ref_time))

    clean = np.stack(projected)
    scale = np.max(np.abs(clean))
    if scale <= 0:
        raise ValueError("Detector projection produced zero strain")
    return clean / scale, np.asarray(antenna, dtype=float), np.asarray(delays, dtype=float)


# ---------------------------------------------------------------------------
# 2-3. colouring and whitening


def design_asd(freq: np.ndarray) -> np.ndarray:
    """aLIGO zero-detuned high-power ASD shape, one at 200 Hz, seismic wall below 10 Hz."""
    import pycbc.psd

    f_min, delta_f = 10.0, 0.25
    n = int(np.ceil(max(float(np.max(freq)), 400.0) / delta_f)) + 2
    psd = pycbc.psd.aLIGOZeroDetHighPower(n, delta_f, f_min)
    grid = psd.sample_frequencies.numpy()
    good = grid >= f_min
    values = np.sqrt(psd.numpy()[good])
    grid = grid[good]
    asd = np.interp(freq, grid, values)
    wall = values[0] * (f_min / np.maximum(freq, 1.0)) ** 3
    asd = np.where(freq < f_min, wall, asd)
    return asd / np.interp(200.0, grid, values)


def colour(white: np.ndarray, sample_rate: float) -> np.ndarray:
    n = white.shape[-1]
    freq = np.fft.rfftfreq(n, 1.0 / sample_rate)
    return np.fft.irfft(np.fft.rfft(white, axis=-1) * design_asd(freq), n=n, axis=-1)


def whiten(raw: np.ndarray, sample_rate: float, cfg: SearchConfig, wdm_cache: WDMCache) -> dict[str, np.ndarray | float]:
    """Per-layer WDM whitening as in ``data_conditioning/whitening.py``.

    The noise RMS of each layer is ``sqrt(0.7191 * median(00^2 + 90^2))``
    over the whole series; the whitened series is the mean of the 00 and 90
    inverse transforms.  Layers outside ``[f_low, f_high]`` are removed.
    """
    wdm = wdm_cache(cfg.whiten_level)
    whitened, rms_all = [], []
    df = 0.0
    for series in raw:
        tf = wdm.t2w(np.asarray(series, dtype=np.float64), sample_rate=sample_rate, t0=0.0, MM=-1)
        coeff = np.asarray(tf.data)
        df = float(tf.df)
        power = coeff.real ** 2 + coeff.imag ** 2
        rms = np.sqrt(NRMS_MEDIAN_FACTOR * np.median(power, axis=1))
        rms = np.maximum(rms, 1e-12)
        layer_freq = np.arange(coeff.shape[0]) * df
        norm = coeff / rms[:, None]
        norm[(layer_freq < cfg.f_low) | (layer_freq > cfg.f_high), :] = 0.0
        tf.data = norm
        ts_00 = np.asarray(wdm.w2t(tf).value, dtype=np.float64)
        ts_90 = np.asarray(wdm.w2tQ(tf).value, dtype=np.float64)
        whitened.append(0.5 * (ts_00 + ts_90)[: series.size])
        rms_all.append(rms)

    return {
        "whitened": np.stack(whitened),
        "freq": np.arange(len(rms_all[0])) * df,
        "rms": np.stack(rms_all),
    }


def display_energy(series: np.ndarray, level: int, sample_rate: float, wdm_cache: WDMCache) -> np.ndarray:
    wdm = wdm_cache(level)
    return np.stack([
        pixel_energy(np.asarray(wdm.t2w(np.asarray(s, dtype=np.float64), sample_rate=sample_rate, t0=0.0, MM=-1).data))
        for s in series
    ])


# ---------------------------------------------------------------------------
# 4-5. multi-resolution maps, pixel selection and clustering


def shifted(series: np.ndarray, k: int) -> np.ndarray:
    """Return x(t + k / fs) as a circular shift."""
    return np.roll(series, -k) if k else series


class FractionalShifter:
    """Band-limited circular shifts x(t + tau) through an FFT phase ramp."""

    def __init__(self, series: np.ndarray, sample_rate: float):
        self.n = series.shape[-1]
        self.spectrum = np.fft.rfft(series, axis=-1)
        self.freq = np.fft.rfftfreq(self.n, 1.0 / sample_rate)

    def __call__(self, tau: float) -> np.ndarray:
        if tau == 0.0:
            return np.fft.irfft(self.spectrum, n=self.n, axis=-1)
        return np.fft.irfft(self.spectrum * np.exp(2j * np.pi * self.freq * tau), n=self.n, axis=-1)


def analysis_time_mask(n_time: int, dt: float, cfg: SearchConfig, total: float) -> np.ndarray:
    t = np.arange(n_time) * dt
    return (t >= cfg.edge_seconds) & (t <= total - cfg.edge_seconds)


def noise_time_mask(n_time: int, dt: float, cfg: SearchConfig, total: float, window: tuple[float, float]) -> np.ndarray:
    t = np.arange(n_time) * dt
    guard = 0.25
    outside = (t < window[0] - guard) | (t > window[1] + guard)
    return analysis_time_mask(n_time, dt, cfg, total) & outside


def band_mask(n_freq: int, df: float, cfg: SearchConfig) -> np.ndarray:
    f = np.arange(n_freq) * df
    return (f >= cfg.f_low) & (f <= cfg.f_high)


def build_resolution(
    white: np.ndarray,
    level: int,
    sample_rate: float,
    cfg: SearchConfig,
    wdm_cache: WDMCache,
    max_shift: int,
    total: float,
    window: tuple[float, float],
    threshold: float | None,
) -> ResolutionMaps:
    wdm = wdm_cache(level)
    n_det = white.shape[0]
    coeff = []
    energy_max = []
    dt = df = 0.0
    shifts = np.arange(-max_shift, max_shift + 1, cfg.max_energy_step)
    if 0 not in shifts:
        shifts = np.sort(np.append(shifts, 0))
    for d in range(n_det):
        running = None
        for k in shifts:
            tf = wdm.t2w(shifted(white[d], int(k)), sample_rate=sample_rate, t0=0.0, MM=-1)
            c = np.asarray(tf.data)
            e = pixel_energy(c)
            running = e if running is None else np.maximum(running, e)
            if k == 0:
                coeff.append(c)
                dt, df = float(tf.dt), float(tf.df)
        energy_max.append(running)
    coeff_arr = np.stack(coeff)
    energy_arr = np.stack(energy_max)
    net = energy_arr.sum(axis=0)

    n_freq, n_time = net.shape
    band = band_mask(n_freq, df, cfg)
    live = analysis_time_mask(n_time, dt, cfg, total)
    valid = band[:, None] & live[None, :]
    if threshold is None:
        noise = band[:, None] & noise_time_mask(n_time, dt, cfg, total, window)[None, :]
        threshold = float(np.quantile(net[noise], 1.0 - cfg.bpp))

    core = valid & (net >= threshold)
    neighbours = ndimage.convolve(core.astype(np.int32), np.ones((3, 3), dtype=np.int32), mode="constant") - core
    selected = core & ((net >= 2.0 * threshold) | (neighbours >= 1))
    labels, _ = ndimage.label(selected, structure=np.ones((3, 3), dtype=bool))
    return ResolutionMaps(level, dt, df, coeff_arr, energy_arr, net, threshold, core, selected, labels)


def collect_fragments(resolutions: list[ResolutionMaps]) -> list[Fragment]:
    fragments = []
    for r, res in enumerate(resolutions):
        if not np.any(res.labels):
            continue
        objects = ndimage.find_objects(res.labels)
        for label, sl in enumerate(objects, start=1):
            if sl is None:
                continue
            sub = res.labels[sl] == label
            f_idx, t_idx = np.nonzero(sub)
            fragments.append(Fragment(r, label, f_idx + sl[0].start, t_idx + sl[1].start))
    return fragments


def fragment_geometry(fragment: Fragment, resolutions: list[ResolutionMaps]) -> tuple[np.ndarray, np.ndarray, float, float]:
    res = resolutions[fragment.res]
    return fragment.t_idx * res.dt, fragment.f_idx * res.df, res.dt, res.df


def link_fragments(fragments: list[Fragment], resolutions: list[ResolutionMaps], cfg: SearchConfig) -> list[tuple[int, int]]:
    """TF-gap links between fragments (``super_cluster_native/utils.get_cluster_links``)."""
    geometry = [fragment_geometry(fr, resolutions) for fr in fragments]
    boxes = [
        (t.min() - 0.5 * dt, t.max() + 0.5 * dt, f.min() - 0.5 * df, f.max() + 0.5 * df)
        for t, f, dt, df in geometry
    ]
    links = []
    for a in range(len(fragments)):
        ta, fa, dta, dfa = geometry[a]
        for b in range(a + 1, len(fragments)):
            tb, fb, dtb, dfb = geometry[b]
            ratio = max(dta, dtb) / min(dta, dtb)
            if ratio > 3.0:
                continue
            dt_min, df_min = min(dta, dtb), min(dfa, dfb)
            box_dt = max(boxes[b][0] - boxes[a][1], boxes[a][0] - boxes[b][1], 0.0)
            box_df = max(boxes[b][2] - boxes[a][3], boxes[a][2] - boxes[b][3], 0.0)
            if box_dt / dt_min + box_df / df_min > cfg.tf_gap:
                continue
            gap_t = np.abs(ta[:, None] - tb[None, :]) - 0.5 * (dta + dtb)
            gap_f = np.abs(fa[:, None] - fb[None, :]) - 0.5 * (dfa + dfb)
            metric = np.maximum(gap_t, 0.0) / dt_min + np.maximum(gap_f, 0.0) / df_min
            if float(metric.min()) <= cfg.tf_gap:
                links.append((a, b))
    return links


def union_find(n: int, links: list[tuple[int, int]]) -> np.ndarray:
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in links:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n)])


def build_superclusters(
    fragments: list[Fragment],
    links: list[tuple[int, int]],
    resolutions: list[ResolutionMaps],
    cfg: SearchConfig,
    noise_mean: list[np.ndarray],
) -> list[Supercluster]:
    """Linked fragments -> size and sub-network cuts -> defragment.

    This is the pattern = 0 order of ``super_cluster_native``: the subnet cut
    runs before defragmentation, so noise fragments that fail it are not
    swallowed by a nearby signal.
    """
    if not fragments:
        return []
    roots = union_find(len(fragments), links)
    groups = [list(np.nonzero(roots == root)[0]) for root in np.unique(roots)]
    candidates = [make_supercluster(g, fragments, resolutions, cfg, noise_mean) for g in groups]
    survivors = [sc.fragments for sc in candidates if sc.passed]
    rejected = [sc for sc in candidates if not sc.passed]

    # defragment: merge survivors whose TF boxes are within (Tgap, Fgap)
    def box(group):
        t = np.concatenate([fragment_geometry(fragments[i], resolutions)[0] for i in group])
        f = np.concatenate([fragment_geometry(fragments[i], resolutions)[1] for i in group])
        return t.min(), t.max(), f.min(), f.max()

    merged = True
    while merged and len(survivors) > 1:
        merged = False
        boxes = [box(g) for g in survivors]
        for a in range(len(survivors)):
            for b in range(a + 1, len(survivors)):
                gap_t = max(boxes[b][0] - boxes[a][1], boxes[a][0] - boxes[b][1], 0.0)
                gap_f = max(boxes[b][2] - boxes[a][3], boxes[a][2] - boxes[b][3], 0.0)
                if gap_t <= cfg.t_gap and gap_f <= cfg.f_gap:
                    survivors[a] = survivors[a] + survivors[b]
                    del survivors[b]
                    merged = True
                    break
            if merged:
                break

    accepted = [make_supercluster(g, fragments, resolutions, cfg, noise_mean) for g in survivors]
    return accepted + rejected


def make_supercluster(
    group: list[int],
    fragments: list[Fragment],
    resolutions: list[ResolutionMaps],
    cfg: SearchConfig,
    noise_mean: list[np.ndarray],
) -> Supercluster:
    res_idx = np.concatenate([np.full(fragments[i].f_idx.size, fragments[i].res) for i in group])
    f_idx = np.concatenate([fragments[i].f_idx for i in group])
    t_idx = np.concatenate([fragments[i].t_idx for i in group])
    dt = np.array([resolutions[r].dt for r in res_idx])
    df = np.array([resolutions[r].df for r in res_idx])
    excess = np.zeros(resolutions[0].coeff.shape[0])
    for r in np.unique(res_idx):
        sel = res_idx == r
        e = resolutions[r].energy_max[:, f_idx[sel], t_idx[sel]]
        excess += np.sum(e - noise_mean[r][:, None], axis=1)
    excess = np.maximum(excess, 0.0)
    total = float(np.sum(excess))
    subnet = float((total - np.max(excess)) / total) if total > 0 else 0.0
    return Supercluster(
        fragments=[int(i) for i in group],
        res=res_idx, f_idx=f_idx, t_idx=t_idx,
        time=t_idx * dt, freq=f_idx * df, dt=dt, df=df,
        det_excess=excess, subnet=subnet,
        passed=bool(res_idx.size >= cfg.min_pixels and subnet >= cfg.subnet_cut),
    )


# ---------------------------------------------------------------------------
# 6-7. time-delay amplitudes and the sky loop


def sky_grid(cfg: SearchConfig, window_length: float) -> dict[str, np.ndarray]:
    import healpy as hp

    npix = hp.nside2npix(cfg.nside)
    theta, phi = hp.pix2ang(cfg.nside, np.arange(npix))
    ra = phi
    dec = 0.5 * np.pi - theta
    ref_time = reference_gps(cfg, window_length)
    fplus, fcross, delay = [], [], []
    for det in detectors(cfg):
        fp, fc = det.antenna_pattern(ra, dec, 0.0, ref_time)
        fplus.append(fp)
        fcross.append(fc)
        delay.append(det.time_delay_from_earth_center(ra, dec, ref_time))
    delay = np.asarray(delay)
    return {
        "ra": ra, "dec": dec,
        "fplus": np.asarray(fplus), "fcross": np.asarray(fcross),
        "rel_delay": delay - delay[0],
    }


def td_amplitudes(
    white: np.ndarray,
    sc: Supercluster,
    resolutions: list[ResolutionMaps],
    td_rate: float,
    sample_rate: float,
    wdm_cache: WDMCache,
    max_shift: int,
) -> np.ndarray:
    """Amplitudes x_d(t + k / td_rate) of every supercluster pixel, k = -K..K.

    ``td_rate`` is finer than the sample rate (cWB ``upTDF``); the shifted
    series are band-limited FFT shifts.  Shape ``(n_det, 2K + 1, n_pixel)``
    complex (real: 00 phase, imag: 90).  Delays are relative to the first
    detector, which therefore only needs k = 0.
    """
    n_det = white.shape[0]
    shifts = np.arange(-max_shift, max_shift + 1)
    out = np.zeros((n_det, shifts.size, sc.res.size), dtype=np.complex128)
    levels = np.unique(sc.res)
    for d in range(n_det):
        if d == 0:
            for r in levels:
                sel = np.nonzero(sc.res == r)[0]
                out[d, :, sel] = resolutions[r].coeff[d, sc.f_idx[sel], sc.t_idx[sel]][:, None]
            continue
        shifter = FractionalShifter(white[d], sample_rate)
        for j, k in enumerate(shifts):
            series = shifter(k / td_rate)
            for r in levels:
                sel = np.nonzero(sc.res == r)[0]
                tf = wdm_cache(resolutions[r].level).t2w(series, sample_rate=sample_rate, t0=0.0, MM=-1)
                out[d, j, sel] = np.asarray(tf.data)[sc.f_idx[sel], sc.t_idx[sel]]
    return out


def dpf_basis(fplus: np.ndarray, fcross: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dominant polarisation frame unit vectors and the regulated cross weight.

    ``fplus``/``fcross`` have shape (n_det, ...).  Returns ``e_plus``,
    ``e_cross`` (same shape) and ``g`` (broadcast shape) where ``g`` scales
    the cross-polarisation projection.  A two-detector network gets the hard
    constraint ``g = 0``: fitting both polarisations there would absorb any
    data and leave no null stream.  Larger networks use
    ``g = |fx|^2 / (|fx|^2 + delta |f+|^2)``.
    """
    n_det = fplus.shape[0]
    f = fplus / np.sqrt(n_det)
    c = fcross / np.sqrt(n_det)
    ff = np.sum(f * f, axis=0)
    cc = np.sum(c * c, axis=0)
    fc = np.sum(f * c, axis=0)
    psi = 0.5 * np.arctan2(2.0 * fc, ff - cc)
    fp = np.cos(psi) * f + np.sin(psi) * c
    fx = -np.sin(psi) * f + np.cos(psi) * c
    norm_p = np.sqrt(np.sum(fp * fp, axis=0))
    norm_x = np.sqrt(np.sum(fx * fx, axis=0))
    e_plus = fp / np.maximum(norm_p, 1e-12)
    e_cross = fx / np.maximum(norm_x, 1e-12)
    if n_det == 2:
        g = np.zeros_like(norm_x)
    else:
        g = norm_x ** 2 / (norm_x ** 2 + delta * norm_p ** 2 + 1e-12)
    return e_plus, e_cross, g


def pixel_likelihood_terms(
    x: np.ndarray, e_plus: np.ndarray, e_cross: np.ndarray, g: np.ndarray, en: float
) -> dict[str, np.ndarray]:
    """Per-pixel energies for data ``x`` of shape (n_det, ..., n_pixel).

    ``e_plus``/``e_cross``/``g`` carry shape (n_det, ..., 1) / (..., 1).
    Energies use the (00^2 + 90^2) / 2 convention and are masked to pixels
    with network energy above ``en``.
    """
    energy = 0.5 * np.sum(x.real ** 2 + x.imag ** 2, axis=0)
    mask = energy > en
    a = np.sum(e_plus * x, axis=0)
    b = np.sum(e_cross * x, axis=0)
    signal = e_plus * a + g * e_cross * b
    null = 0.5 * np.sum(np.abs(x - signal) ** 2, axis=0)
    # x^T P x splits into the diagonal (incoherent) and cross-detector
    # (coherent) terms of the projector P = e+ e+^T + g ex ex^T.
    projected = 0.5 * (np.abs(a) ** 2 + g * np.abs(b) ** 2)
    incoherent = 0.5 * np.sum(np.abs(x) ** 2 * (e_plus ** 2 + g * e_cross ** 2), axis=0)
    coherent = projected - incoherent
    return {
        "energy": np.where(mask, energy, 0.0),
        "null": np.where(mask, null, 0.0),
        "likelihood": np.where(mask, energy - null, 0.0),
        "coherent": np.where(mask, coherent, 0.0),
        "signal": signal,
        "mask": mask,
    }


def sky_loop(
    amp: np.ndarray,
    sky: dict[str, np.ndarray],
    td_rate: float,
    max_shift: int,
    cfg: SearchConfig,
    chunk: int = 256,
) -> SkyScan:
    n_det, _, n_pix = amp.shape
    n_sky = sky["ra"].size
    shift = np.clip(np.rint(sky["rel_delay"] * td_rate).astype(int), -max_shift, max_shift)
    en = cfg.acore ** 2 * n_det
    e_plus, e_cross, g = dpf_basis(sky["fplus"], sky["fcross"], cfg.delta)
    sums = {key: np.zeros(n_sky) for key in ("energy", "likelihood", "coherent", "null")}
    for start in range(0, n_sky, chunk):
        stop = min(start + chunk, n_sky)
        x = np.stack([amp[d, shift[d, start:stop] + max_shift, :] for d in range(n_det)])
        terms = pixel_likelihood_terms(
            x, e_plus[:, start:stop, None], e_cross[:, start:stop, None], g[start:stop, None], en
        )
        for key in sums:
            sums[key][start:stop] = np.sum(terms[key], axis=-1)
    net_cc = sums["coherent"] / (np.abs(sums["coherent"]) + sums["null"] + 1e-12)
    statistic = sums["likelihood"] * net_cc
    best = int(np.argmax(statistic))
    return SkyScan(
        ra=sky["ra"], dec=sky["dec"], fplus=sky["fplus"], fcross=sky["fcross"],
        rel_delay=sky["rel_delay"], shift=shift,
        energy=sums["energy"], likelihood=sums["likelihood"], coherent=sums["coherent"],
        null=sums["null"], net_cc=net_cc, statistic=statistic, best=best,
    )


def terms_at_sky(amp: np.ndarray, scan: SkyScan, index: int, max_shift: int, cfg: SearchConfig) -> dict[str, np.ndarray]:
    n_det = amp.shape[0]
    x = np.stack([amp[d, scan.shift[d, index] + max_shift, :] for d in range(n_det)])
    e_plus, e_cross, g = dpf_basis(scan.fplus[:, index : index + 1], scan.fcross[:, index : index + 1], cfg.delta)
    terms = pixel_likelihood_terms(x, e_plus, e_cross, g, cfg.acore ** 2 * n_det)
    terms["x"] = x
    return terms


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: list[float]) -> np.ndarray:
    order = np.argsort(values)
    cdf = np.cumsum(weights[order])
    return np.interp(np.asarray(quantiles) * cdf[-1], cdf, values[order])


def make_trigger(
    index: int, sc: Supercluster, amp: np.ndarray, scan: SkyScan, max_shift: int, cfg: SearchConfig
) -> Trigger:
    terms = terms_at_sky(amp, scan, scan.best, max_shift, cfg)
    per_res = np.array([np.sum(terms["energy"][sc.res == r]) for r in np.unique(sc.res)])
    norm = max(float(np.sum(per_res) / max(np.max(per_res), 1e-12)), 1.0)
    energy = scan.energy[scan.best] / norm
    coherent = scan.coherent[scan.best] / norm
    null = scan.null[scan.best] / norm
    net_cc = float(scan.net_cc[scan.best])
    n_det = amp.shape[0]
    rho = float(np.sqrt(max(coherent * net_cc, 0.0) / max(n_det - 1, 1)))
    weights = np.maximum(terms["likelihood"], 0.0)
    if np.sum(weights) > 0:
        time = float(np.sum(weights * sc.time) / np.sum(weights))
        f_low, f_high = weighted_quantile(sc.freq, weights, [0.02, 0.98])
    else:
        time = float(np.mean(sc.time))
        f_low, f_high = float(np.min(sc.freq)), float(np.max(sc.freq))
    return Trigger(
        cluster=index, time=time, f_low=float(f_low), f_high=float(f_high),
        ra=float(scan.ra[scan.best]), dec=float(scan.dec[scan.best]), norm=norm,
        energy=float(energy), likelihood=float(scan.likelihood[scan.best] / norm),
        coherent=float(coherent), null=float(null), net_cc=net_cc, rho=rho,
        rel_delay=scan.rel_delay[:, scan.best].copy(),
        passed=bool(rho >= cfg.net_rho_cut and net_cc >= cfg.net_cc_cut),
    )


# ---------------------------------------------------------------------------
# one lag of the search


def analyse_lag(
    white: np.ndarray,
    lag: float,
    sample_rate: float,
    cfg: SearchConfig,
    wdm_cache: WDMCache,
    sky: dict[str, np.ndarray],
    window: tuple[float, float],
    thresholds: list[float] | None = None,
    noise_mean: list[np.ndarray] | None = None,
) -> LagAnalysis:
    total = white.shape[-1] / sample_rate
    max_shift = int(np.ceil(max_light_travel_time(cfg) * sample_rate)) + 1
    td_rate = sample_rate * cfg.td_upsample
    td_shift = int(np.ceil(max_light_travel_time(cfg) * td_rate)) + 1
    resolutions = [
        build_resolution(
            white, level, sample_rate, cfg, wdm_cache, max_shift, total, window,
            None if thresholds is None else thresholds[i],
        )
        for i, level in enumerate(cfg.levels)
    ]
    if noise_mean is None:
        noise_mean = []
        for res in resolutions:
            noise = band_mask(res.n_freq, res.df, cfg)[:, None] & noise_time_mask(res.n_time, res.dt, cfg, total, window)[None, :]
            noise_mean.append(np.array([np.mean(e[noise]) for e in res.energy_max]))

    fragments = collect_fragments(resolutions)
    links = link_fragments(fragments, resolutions, cfg)
    superclusters = build_superclusters(fragments, links, resolutions, cfg, noise_mean)

    triggers, scans, amplitudes = [], {}, {}
    for i, sc in enumerate(superclusters):
        if not sc.passed:
            continue
        amp = td_amplitudes(white, sc, resolutions, td_rate, sample_rate, wdm_cache, td_shift)
        scan = sky_loop(amp, sky, td_rate, td_shift, cfg)
        scans[i] = scan
        amplitudes[i] = amp
        triggers.append(make_trigger(i, sc, amp, scan, td_shift, cfg))

    return LagAnalysis(lag, resolutions, fragments, links, superclusters, triggers, scans, amplitudes, noise_mean)


# ---------------------------------------------------------------------------
# reconstruction


def reconstruct(
    white: np.ndarray,
    analysis: LagAnalysis,
    trigger: Trigger,
    sample_rate: float,
    cfg: SearchConfig,
    wdm_cache: WDMCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Whitened waveform per detector plus reconstructed / null TF energy.

    Each resolution's signal pixels are inverted (mean of the 00 and 90
    inverses), shifted back from the aligned frame into each detector, and
    the resolutions are averaged with weights set by their signal energy.
    """
    sc = analysis.superclusters[trigger.cluster]
    amp = analysis.td_amplitudes[trigger.cluster]
    scan = analysis.sky[trigger.cluster]
    max_shift = (amp.shape[1] - 1) // 2
    terms = terms_at_sky(amp, scan, scan.best, max_shift, cfg)
    signal = np.where(terms["mask"][None, :], terms["signal"], 0.0)
    n_det, n_samples = white.shape

    recon = np.zeros_like(white)
    weight_sum = 0.0
    display = cfg.levels.index(cfg.display_level) if cfg.display_level in cfg.levels else 0
    res_display = analysis.resolutions[display]
    recon_energy = np.zeros((res_display.n_freq, res_display.n_time))
    null_energy = np.zeros_like(recon_energy)
    for r in np.unique(sc.res):
        sel = np.nonzero(sc.res == r)[0]
        res = analysis.resolutions[r]
        wdm = wdm_cache(res.level)
        weight = float(np.sum(pixel_energy(signal[:, sel])))
        if weight <= 0:
            continue
        template = wdm.t2w(np.zeros(n_samples), sample_rate=sample_rate, t0=0.0, MM=-1)
        for d in range(n_det):
            data = np.zeros((res.n_freq, res.n_time), dtype=np.complex128)
            data[sc.f_idx[sel], sc.t_idx[sel]] = signal[d, sel]
            template.data = data
            wave = 0.5 * (np.asarray(wdm.w2t(template).value) + np.asarray(wdm.w2tQ(template).value))
            tau = scan.shift[d, scan.best] / (sample_rate * cfg.td_upsample)
            recon[d] += weight * FractionalShifter(wave[:n_samples], sample_rate)(-tau)
        weight_sum += weight
        if r == display:
            recon_energy[sc.f_idx[sel], sc.t_idx[sel]] = np.sum(pixel_energy(signal[:, sel]), axis=0)
            null_energy[sc.f_idx[sel], sc.t_idx[sel]] = np.sum(pixel_energy(terms["x"][:, sel] - signal[:, sel]), axis=0)
    if weight_sum > 0:
        recon /= weight_sum
    return recon, recon_energy, null_energy


def sky_pixel_map(analysis: LagAnalysis, trigger: Trigger, sky_index: int, cfg: SearchConfig) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel coherent and null energy at the display resolution for one sky point."""
    sc = analysis.superclusters[trigger.cluster]
    amp = analysis.td_amplitudes[trigger.cluster]
    scan = analysis.sky[trigger.cluster]
    max_shift = (amp.shape[1] - 1) // 2
    display = cfg.levels.index(cfg.display_level)
    res = analysis.resolutions[display]
    sel = sc.res == display
    coherent = np.zeros((res.n_freq, res.n_time))
    null = np.zeros_like(coherent)
    if np.any(sel):
        terms = terms_at_sky(amp[:, :, sel], scan, sky_index, max_shift, cfg)
        coherent[sc.f_idx[sel], sc.t_idx[sel]] = terms["coherent"]
        null[sc.f_idx[sel], sc.t_idx[sel]] = terms["null"]
    return coherent, null


# ---------------------------------------------------------------------------
# full run


def run_search(input_path: Path, cfg: SearchConfig, progress=print) -> SearchResult:
    import healpy as hp

    hplus, sample_rate = load_logo_waveform(input_path)
    clean, antenna, arrival = project_to_detectors(hplus, sample_rate, cfg)
    window_length = hplus.size / sample_rate
    pad = int(round(cfg.pad_seconds * sample_rate))
    n_total = hplus.size + 2 * pad
    window = (cfg.pad_seconds, cfg.pad_seconds + window_length)
    time = np.arange(n_total) / sample_rate

    clean_full = np.zeros((len(cfg.ifos), n_total))
    clean_full[:, pad : pad + hplus.size] = clean
    white_clean = clean_full / cfg.noise_sigma
    rng = np.random.default_rng(cfg.noise_seed)
    white_true = white_clean + rng.normal(0.0, 1.0, size=white_clean.shape)
    raw = colour(white_true, sample_rate)

    wdm_cache = WDMCache(cfg)
    progress("  whitening")
    cond = whiten(raw, sample_rate, cfg, wdm_cache)
    whitened = cond["whitened"]
    asd_at_layers = design_asd(np.maximum(cond["freq"], 1.0))

    sky = sky_grid(cfg, window_length)
    progress("  zero-lag search")
    zero = analyse_lag(whitened, 0.0, sample_rate, cfg, wdm_cache, sky, window)
    if not zero.triggers:
        raise RuntimeError("Zero-lag search produced no trigger; lower --bpp or --noise-sigma")
    thresholds = [res.threshold for res in zero.resolutions]
    noise_mean = zero.noise_mean

    in_window = [t for t in zero.triggers if window[0] <= t.time <= window[1]]
    event = max(in_window or zero.triggers, key=lambda t: zero.sky[t.cluster].statistic[zero.sky[t.cluster].best])
    recon, recon_energy, null_energy = reconstruct(whitened, zero, event, sample_rate, cfg, wdm_cache)

    lags = np.arange(1, cfg.n_lags + 1) * cfg.lag_step
    background = []
    for i, lag in enumerate(lags):
        shift = int(round(lag * sample_rate))
        lagged = whitened.copy()
        for d in range(1, lagged.shape[0]):
            lagged[d] = np.roll(lagged[d], d * shift)
        result = analyse_lag(lagged, float(lag), sample_rate, cfg, wdm_cache, sky, window, thresholds, noise_mean)
        background.append(result.triggers)
        if (i + 1) % max(1, len(lags) // 5) == 0 or i + 1 == len(lags):
            progress(f"  time slides {i + 1}/{len(lags)}")

    source_index = int(hp.ang2pix(cfg.nside, 0.5 * np.pi - cfg.source_dec, cfg.source_ra))
    return SearchResult(
        config=cfg, sample_rate=sample_rate, time=time, window=window,
        hplus=hplus, antenna=antenna, arrival_delay=arrival,
        clean=clean_full, white_clean=white_clean, raw=raw, whitened=whitened,
        white_freq=cond["freq"], white_rms=cond["rms"], asd_at_layers=asd_at_layers,
        raw_display=display_energy(raw, cfg.display_level, sample_rate, wdm_cache),
        white_display=display_energy(whitened, cfg.display_level, sample_rate, wdm_cache),
        zero_lag=zero, event=event, reconstructed=recon,
        recon_energy=recon_energy, null_energy=null_energy,
        source_sky_index=source_index, lags=lags, background=background,
    )
