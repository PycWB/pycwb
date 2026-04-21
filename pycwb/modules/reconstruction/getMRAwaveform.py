"""
MRA waveform reconstruction with optional Numba JIT acceleration.

When numba is available, the per-pixel base-wave computation and accumulation
loop runs entirely inside @njit-compiled code.  When rocket-fft is also
installed, the time-of-flight phase-shift correction is JIT-compiled too.
"""
import logging
import numpy as np
from dataclasses import dataclass

from pycwb.types.time_series import TimeSeries

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional Numba / rocket-fft availability
# ---------------------------------------------------------------------------
try:
    from numba import njit as _numba_njit
    import numba

    HAS_NUMBA = True
    try:
        import rocket_fft as _rocket_fft  # noqa: F401
        HAS_ROCKET_FFT = True
    except ImportError:
        HAS_ROCKET_FFT = False
except ImportError:

    def _numba_njit(*args, **kwargs):
        """No-op decorator when numba is unavailable."""
        def decorator(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    HAS_NUMBA = False
    HAS_ROCKET_FFT = False

# ===================================================================
# WDM kernel setup (pure Python – one-time cost)
# ===================================================================


@dataclass
class _PyWDMKernel:
    max_layer: int
    m_H: int
    wavelet_filter: np.ndarray


def _extract_wdm_filter_and_mh(wdm_obj):
    wavelet_filter = None
    for attr in ("wavelet_filter", "wdmFilter", "wdm_filter", "filter"):
        if hasattr(wdm_obj, attr):
            value = getattr(wdm_obj, attr)
            if value is not None:
                wavelet_filter = np.asarray(value, dtype=np.float64)
                break

    if wavelet_filter is None and hasattr(wdm_obj, "get_wavelet_filter"):
        wavelet_filter = np.asarray(wdm_obj.get_wavelet_filter(), dtype=np.float64)

    if wavelet_filter is None:
        raise ValueError("Cannot extract wavelet filter from WDM object")

    m_h = None
    for attr in ("m_H", "m_h", "filter_len", "n_filter_taps"):
        if hasattr(wdm_obj, attr):
            value = getattr(wdm_obj, attr)
            if value is not None:
                m_h = int(value)
                break
    if m_h is None:
        m_h = int(len(wavelet_filter))

    return wavelet_filter, m_h


_wdm_set_cache: dict[tuple, list] = {}  # keyed by (l_low, l_high, rateANA, segEdge, TDSize, beta_order, precision)


def _create_wdm_set_python(config):
    from wdm_wavelet.wdm import WDM as WDMWavelet

    l_low = int(config.l_low)
    l_high = int(config.l_high)
    rate_ana = float(config.rateANA)
    seg_edge = float(config.segEdge)
    td_size = int(config.TDSize)
    beta_order = int(getattr(config, "WDM_beta_order", 6))
    precision = int(getattr(config, "WDM_precision", 10))

    cache_key = (l_low, l_high, rate_ana, seg_edge, td_size, beta_order, precision)
    if cache_key in _wdm_set_cache:
        return _wdm_set_cache[cache_key]

    wdm_list = []
    for i in range(l_low, l_high + 1):
        level = l_high + l_low - i
        layers = max(1, 2 ** level)

        wdm = WDMWavelet(M=layers, K=layers, beta_order=beta_order, precision=precision)
        wavelet_filter, m_h = _extract_wdm_filter_and_mh(wdm)

        wdm_f_len = float(m_h) / rate_ana
        if wdm_f_len > seg_edge + 0.001:
            raise ValueError(
                f"Filter length must be <= segEdge (filter={wdm_f_len} sec, segEdge={seg_edge} sec)"
            )

        rate = max(1.0, rate_ana / float(layers))
        if seg_edge < int(1.5 * (td_size / rate) + 0.5):
            raise ValueError(
                "segEdge must be > 1.5x the length for time delay amplitudes"
            )

        wdm_list.append(
            _PyWDMKernel(
                max_layer=layers,
                m_H=m_h,
                wavelet_filter=wavelet_filter,
            )
        )

    _wdm_set_cache[cache_key] = wdm_list
    return wdm_list


# ===================================================================
# njit base-wave kernels
# ===================================================================

@_numba_njit(cache=True)
def _get_base_wave_njit(wdm_filter, max_layer, m, n):
    """00-phase base wavelet vector for pixel (m, n)."""
    N = len(wdm_filter)
    M = max_layer
    wlen = 2 * (N - 1) + 1
    w = np.zeros(wlen, dtype=np.float64)

    if m == 0:
        if n % 2 == 0:
            for k in range(N):
                w[N - 1 + k] = wdm_filter[k]
                w[N - 1 - k] = wdm_filter[k]
    elif m == M:
        if (n - M) % 2 == 0:
            s = -1.0 if (M % 2) else 1.0
            w[N - 1] = s * wdm_filter[0]
            for k in range(1, N):
                s = -s
                w[N - 1 + k] = s * wdm_filter[k]
                w[N - 1 - k] = s * wdm_filter[k]
    else:
        ratio = m * np.pi / M
        sqrt2 = np.sqrt(2.0)
        if (m + n) % 2:
            for k in range(1, N):
                val = -sqrt2 * np.sin(k * ratio) * wdm_filter[k]
                w[N - 1 + k] = val
                w[N - 1 - k] = -val
        else:
            w[N - 1] = sqrt2 * wdm_filter[0]
            for k in range(1, N):
                val = sqrt2 * np.cos(k * ratio) * wdm_filter[k]
                w[N - 1 + k] = val
                w[N - 1 - k] = val

    return n * M - (N - 1), w


@_numba_njit(cache=True)
def _get_base_wave_quad_njit(wdm_filter, max_layer, m, n):
    """90-phase (quadrature) base wavelet vector for pixel (m, n)."""
    N = len(wdm_filter)
    M = max_layer
    wlen = 2 * (N - 1) + 1
    w = np.zeros(wlen, dtype=np.float64)

    if m == 0:
        if n % 2 == 1:
            for k in range(N):
                w[N - 1 + k] = wdm_filter[k]
                w[N - 1 - k] = wdm_filter[k]
    elif m == M:
        if (n - M) % 2 == 1:
            s = 1.0
            w[N - 1] = wdm_filter[0]
            for k in range(1, N):
                s = -s
                w[N - 1 + k] = s * wdm_filter[k]
                w[N - 1 - k] = s * wdm_filter[k]
    else:
        ratio = m * np.pi / M
        sqrt2 = np.sqrt(2.0)
        if (m + n) % 2:
            w[N - 1] = sqrt2 * wdm_filter[0]
            for k in range(1, N):
                val = sqrt2 * np.cos(k * ratio) * wdm_filter[k]
                w[N - 1 + k] = val
                w[N - 1 - k] = val
        else:
            for k in range(1, N):
                val = sqrt2 * np.sin(k * ratio) * wdm_filter[k]
                w[N - 1 + k] = val
                w[N - 1 - k] = -val

    return n * M - (N - 1), w


@_numba_njit(cache=True)
def _get_base_wave_dispatch(max_layer, wdm_filter, tf_index, quad):
    """Dispatch to 00/90-phase base wave by (m, n) from tf_index."""
    M1 = max_layer + 1
    m = tf_index % M1
    n = tf_index // M1
    if quad:
        return _get_base_wave_quad_njit(wdm_filter, max_layer, m, n)
    else:
        return _get_base_wave_njit(wdm_filter, max_layer, m, n)


# ===================================================================
# SOA pixel extraction (Python → NumPy arrays for njit)
# ===================================================================

def _extract_pixel_arrays(pixels, nIFO):
    """Convert list[Pixel] to flat NumPy arrays (structure-of-arrays)."""
    n_pix = len(pixels)

    pix_time = np.empty(n_pix, dtype=np.int64)
    pix_layers = np.empty(n_pix, dtype=np.int64)
    pix_rate = np.empty(n_pix, dtype=np.float64)
    pix_core = np.empty(n_pix, dtype=np.int64)
    pix_noise_rms = np.empty((n_pix, nIFO), dtype=np.float64)
    pix_wave = np.empty((n_pix, nIFO), dtype=np.float64)
    pix_w90 = np.empty((n_pix, nIFO), dtype=np.float64)
    pix_asnr = np.empty((n_pix, nIFO), dtype=np.float64)
    pix_a90 = np.empty((n_pix, nIFO), dtype=np.float64)

    for p, pix in enumerate(pixels):
        pix_time[p] = pix.time
        pix_layers[p] = pix.layers
        pix_rate[p] = pix.rate
        pix_core[p] = pix.core
        for d in range(nIFO):
            pd = pix.data[d]
            pix_noise_rms[p, d] = pd.noise_rms
            pix_wave[p, d] = pd.wave
            pix_w90[p, d] = pd.w_90
            pix_asnr[p, d] = pd.asnr
            pix_a90[p, d] = pd.a_90

    return (pix_time, pix_layers, pix_rate, pix_core,
            pix_noise_rms, pix_wave, pix_w90, pix_asnr, pix_a90)


def _pa_to_tuple(pa):
    """Extract the same tuple layout as ``_extract_pixel_arrays`` directly from a
    :class:`~pycwb.types.pixel_arrays.PixelArrays` — zero Python iteration."""
    pix_time   = pa.time.astype(np.int64)
    pix_layers = pa.layers.astype(np.int64)
    pix_rate   = pa.rate.astype(np.float64)
    pix_core   = pa.core.astype(np.int64)
    # pixel_arrays uses (n_ifo, n_pix); kernel expects (n_pix, n_ifo)
    pix_noise_rms = pa.noise_rms.T.astype(np.float64)
    pix_wave      = pa.wave.T.astype(np.float64)
    pix_w90       = pa.w_90.T.astype(np.float64)
    pix_asnr      = pa.asnr.T.astype(np.float64)
    pix_a90       = pa.a_90.T.astype(np.float64)
    return (pix_time, pix_layers, pix_rate, pix_core,
            pix_noise_rms, pix_wave, pix_w90, pix_asnr, pix_a90)


def _build_wdm_njit_data(wdm_list):
    """Build flat arrays for WDM lookup inside njit."""
    n_wdm = len(wdm_list)
    wdm_keys = np.empty(n_wdm, dtype=np.int64)
    wdm_max_layers = np.empty(n_wdm, dtype=np.int64)

    if HAS_NUMBA:
        wdm_filters = numba.typed.List()
    else:
        wdm_filters = []

    for i, w in enumerate(wdm_list):
        wdm_keys[i] = int(w.max_layer) + 1
        wdm_max_layers[i] = int(w.max_layer)
        wdm_filters.append(w.wavelet_filter)

    return wdm_keys, wdm_max_layers, wdm_filters


# ===================================================================
# Core njit MRA accumulation kernel
# ===================================================================

@_numba_njit(cache=True)
def _mra_wave_kernel(
    pix_time,        # int64   (n_pix,)
    pix_layers,      # int64   (n_pix,)
    pix_core,        # int64   (n_pix,)
    pix_a00,         # float64 (n_pix,) – pre-selected & scaled 00-phase amplitude
    pix_a90,         # float64 (n_pix,) – pre-selected & scaled 90-phase amplitude
    wdm_keys,        # int64   (n_wdm,) – lookup keys (= max_layer + 1)
    wdm_max_layers,  # int64   (n_wdm,)
    wdm_filters,     # typed list of float64 1-D arrays
    mode,            # int
    io,              # int – index offset
    z_len,           # int – output length
):
    """Accumulate all pixel base-wave contributions into output array."""
    z = np.zeros(z_len, dtype=np.float64)
    n_pix = pix_time.shape[0]
    n_wdm = wdm_keys.shape[0]

    for p in range(n_pix):
        if pix_core[p] == 0:
            continue

        # lookup WDM filter for this pixel's resolution
        layers_key = pix_layers[p]
        wdm_idx = -1
        for k in range(n_wdm):
            if wdm_keys[k] == layers_key:
                wdm_idx = k
                break
        if wdm_idx < 0:
            continue

        max_layer = wdm_max_layers[wdm_idx]
        wdm_filter = wdm_filters[wdm_idx]

        a00 = pix_a00[p]
        a90 = pix_a90[p]
        if mode < 0:
            a00 = 0.0
        if mode > 0:
            a90 = 0.0

        tf_index = pix_time[p]

        # 00-phase contribution
        if a00 != 0.0:
            j00, x00 = _get_base_wave_dispatch(max_layer, wdm_filter, tf_index, False)
            j00 -= io
            for i in range(len(x00)):
                idx = j00 + i
                if 0 <= idx < z_len:
                    z[idx] += x00[i] * a00

        # 90-phase contribution
        if a90 != 0.0:
            j90, x90 = _get_base_wave_dispatch(max_layer, wdm_filter, tf_index, True)
            j90 -= io
            for i in range(len(x90)):
                idx = j90 + i
                if 0 <= idx < z_len:
                    z[idx] += x90[i] * a90

    return z


# ===================================================================
# Time-of-flight phase-shift kernel (requires rocket-fft for njit)
# ===================================================================

def _apply_tof_correction(z_data, dt, t_shift):
    """Frequency-domain phase shift for time-of-flight correction."""
    n = len(z_data)
    xf = np.fft.rfft(z_data)
    freqs = np.fft.rfftfreq(n, d=dt)
    xf *= np.exp(-2j * np.pi * freqs * t_shift)
    return np.fft.irfft(xf, n=n)


if HAS_NUMBA and HAS_ROCKET_FFT:
    _apply_tof_correction = _numba_njit(cache=True)(_apply_tof_correction)


# ===================================================================
# Public API
# ===================================================================

def get_MRA_wave(cluster, wdmList, rate, ifo, a_type, mode, nproc,
                 whiten=False, _pixel_arrays=None, _wdm_njit_data=None) -> TimeSeries | None:
    """
    Get MRA waveforms of type a_type in time domain given lag number and cluster ID.

    Parameters
    ----------
    cluster : pycwb.types.network_cluster.Cluster
        cluster object
    wdmList : list of _PyWDMKernel
        list of WDM kernel objects
    rate : float
        sampling rate
    ifo : int
        IFO id
    a_type : str
        type of waveforms, 'signal' or 'strain'
    mode : int
        -1/0/1 - return 90/mra/0 phase
    nproc : int
        number of processes (unused, kept for API compat)
    whiten : bool
        if True, reconstruct whitened waveforms
    _pixel_arrays : tuple or None
        pre-extracted SOA pixel arrays (internal, for reuse across IFOs)
    _wdm_njit_data : tuple or None
        pre-built (wdm_keys, wdm_max_layers, wdm_filters) for njit kernel

    Returns
    -------
    waveform : pycwb.types.time_series.TimeSeries
        reconstructed waveform
    """
    if not cluster.pixel_arrays:
        return None

    max_f_len = max(wdm.m_H / rate for wdm in wdmList)

    # find event time interval using pixel_arrays directly
    _pa = cluster.pixel_arrays
    T_arr = (_pa.time.astype(np.float64) / _pa.layers.astype(np.float64)) / _pa.rate.astype(np.float64)
    tmin = float(T_arr.min())
    tmax = float(T_arr.max())

    tmin = int(tmin - max_f_len) - 1
    tmax = int(tmax + max_f_len) + 1

    z_len = int(rate * (tmax - tmin) + 0.1)
    dt = 1.0 / rate
    io = int(tmin / dt + 0.01)

    # Build pre-extracted arrays if not provided
    if _pixel_arrays is None:
        nIFO = _pa._n_ifo
        _pixel_arrays = _pa_to_tuple(_pa)
    if _wdm_njit_data is None:
        _wdm_njit_data = _build_wdm_njit_data(wdmList)

    (pix_time, pix_layers, pix_rate, pix_core,
     pix_noise_rms, pix_wave, pix_w90, pix_asnr, pix_a90) = _pixel_arrays

    # Select and scale amplitudes for this IFO
    if a_type == 'signal':
        a00_col = pix_asnr[:, ifo].copy()
        a90_col = pix_a90[:, ifo].copy()
    else:
        a00_col = pix_wave[:, ifo].copy()
        a90_col = pix_w90[:, ifo].copy()

    if not whiten:
        rms_col = pix_noise_rms[:, ifo]
        a00_col *= rms_col
        a90_col *= rms_col

    wdm_keys, wdm_max_layers, wdm_filters = _wdm_njit_data

    z_data = _mra_wave_kernel(
        pix_time, pix_layers, pix_core,
        a00_col, a90_col,
        wdm_keys, wdm_max_layers, wdm_filters,
        mode, io, z_len,
    )

    return TimeSeries(data=z_data, dt=dt, t0=tmin)


def get_network_MRA_wave(config, cluster, rate, nIFO, rTDF, a_type, mode, tof,
                         whiten=False, in_rate=None, start_time=None):
    """
    Get MRA waveforms of type a_type in time domain given lag number and cluster ID.

    Parameters
    ----------
    config : pycwb.config.Config
        configuration object
    cluster : pycwb.types.network_cluster.Cluster
        cluster object
    rate : float
        sampling rate
    nIFO : int
        number of IFOs
    rTDF : float
        effective time-delay
    a_type : str
        type of waveforms, 'signal' or 'strain'
    mode : int
        -1/0/1 - return 90/mra/0 phase
    tof : bool
        if True, apply time-of-flight corrections
    whiten : bool
        if True, reconstruct whitened waveforms
    in_rate : float, optional
        input rate of the original data, used for rescaling the waveforms.
    start_time : float, optional
        if set, shift waveform start times by this value.

    Returns
    -------
    waveforms : list of pycwb.types.time_series.TimeSeries
        reconstructed waveforms
    """
    wdm_list = _create_wdm_set_python(config)

    # Pre-extract pixel data into flat arrays (once, reused for all IFOs)
    pixel_arrays = _pa_to_tuple(cluster.pixel_arrays)
    wdm_njit_data = _build_wdm_njit_data(wdm_list)

    v = cluster.sky_time_delay  # backward time delay per IFO

    waveforms = []
    for i in range(nIFO):
        x = get_MRA_wave(cluster, wdm_list, rate, i, a_type, mode,
                         nproc=config.nproc, whiten=whiten,
                         _pixel_arrays=pixel_arrays, _wdm_njit_data=wdm_njit_data)
        if len(x.data) == 0:
            logger.warning("zero length")
            return False

        if tof:
            R = rTDF
            t_shift = -v[i] / R
            corrected = _apply_tof_correction(x.data, x.dt, t_shift)
            x = TimeSeries(data=corrected, t0=x.t0, dt=x.dt)

        waveforms.append(x)

        if in_rate is not None:
            rescale = 1.0 / np.sqrt(2) ** (np.log2(in_rate / x.sample_rate))
            x.data *= rescale

        if start_time is not None:
            x.start_time += start_time

    return waveforms


# ===================================================================
# Public base-wave entry point (delegates to njit dispatch)
# ===================================================================

def get_base_wave(max_layer: int, wdm_filter: np.ndarray, tf_index: int, quad: bool) -> tuple[int, np.ndarray]:
    """Compute base wavelet vector for a single pixel."""
    return _get_base_wave_dispatch(max_layer, wdm_filter, tf_index, quad)
