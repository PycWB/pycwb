"""
Pure-Python whitening without ROOT dependencies.

This module provides Python-native implementations of cWB whitening
using TimeFrequencyMap methods and NumPy/SciPy operations.
"""

import logging

import numpy as np
from scipy import signal as scipy_signal
from wdm_wavelet.wdm import WDM

logger = logging.getLogger(__name__)


def whitening_python(config, h):
    """
    Noise whitening via WDM (pure-Python implementation).

    This performs time-frequency whitening using the WDM (Wilson-Daubechies-Meyer)
    wavelet decomposition from the wdm_wavelet package.

    **Algorithm:**
    1. Transform data to TF domain using WDM
    2. Estimate noise RMS per frequency-bin using running median
    3. Apply bandpass filter to RMS estimate
    4. Whiten coefficients by dividing by noise RMS
    5. Transform whitened coefficients back to time domain
    6. Also transform RMS map back to time domain for nRMS output

    **Note:** This is a simplified implementation that demonstrates the pure-Python
    approach. The RMS estimation algorithm may differ from the C++ version, leading
    to numerical differences. Further tuning is needed to match C++ exactly.

    Parameters
    ----------
    config : Config
        Configuration object with:
        - rateANA: Analysis sample rate (Hz)
        - WDM_beta_order: Meyer wavelet beta order (default: 6)
        - WDM_precision: WDM precision (default: 10)
        - fLow, fHigh: Frequency band limits
        - whiteWindow: Window for noise estimation (seconds, default: 60)
        - segEdge: Edge length to exclude (seconds)

    h : pycbc.types.timeseries.TimeSeries | gwpy.timeseries.TimeSeries
        Input time-series data (after regression).

    Returns
    -------
    tuple[pycbc.types.timeseries.TimeSeries, pycbc.types.timeseries.TimeSeries]
        `(whitened_strain, nRMS)` - whitened data and noise RMS, both as TimeSeries.
    """
    import pycbc.types

    # Ensure input is pycbc TimeSeries (it should already be from data generation/regression)
    if not isinstance(h, pycbc.types.TimeSeries):
        # Convert if needed (gwpy → pycbc)
        h_ts = pycbc.types.TimeSeries(h.value, delta_t=h.dt.value, epoch=h.t0.value)
    else:
        h_ts = h

    # Get WDM parameters
    layers = int(config.rateANA / 8)
    beta_order = getattr(config, 'WDM_beta_order', 6)
    precision = getattr(config, 'WDM_precision', 10)

    # Get whitening parameters
    white_window = getattr(config, 'whiteWindow', 60.0) if hasattr(config, 'whiteWindow') and config.whiteWindow is not None else 60.0
    edge_length = getattr(config, 'segEdge', 10.0)
    f_low = float(config.fLow)
    f_high = float(config.fHigh)
    sample_rate = float(h_ts.sample_rate)

    logger.info(f"Python whitening: M={layers}, beta={beta_order}, prec={precision}")
    logger.info(f"  Window={white_window}s, Edge={edge_length}s, Band=[{f_low}, {f_high}] Hz")

    # Create WDM transform
    wdm = WDM(M=layers, K=layers, beta_order=beta_order, precision=precision)

    # Transform to TF domain
    # Convert pycbc TimeSeries to numpy for wdm_wavelet
    signal_data = np.array(h_ts.data, dtype=np.float64)
    t0 = float(h_ts.start_time)

    # Forward transform (MM=-1 for quadratures: 00-phase and 90-phase)
    # Note: first argument is the signal itself (positional), not keyword
    tf_map = wdm.t2w(signal_data, sample_rate=sample_rate, t0=t0, MM=-1)

    logger.info(f"  TF map shape: {tf_map.data.shape} (complex)")

    # Estimate noise using cWB algorithm (percentile-based)
    median_map, norm50_map = _estimate_noise_rms_cwb(
        tf_map,
        original_length=len(signal_data),  # Pass original audio length
        window_length=white_window,
        stride=white_window,  # Match C++ default (stride=window)
        edge_length=edge_length,
        sample_rate=sample_rate
    )

    logger.info(f"  Noise median shape: {median_map.shape}")
    logger.info(f"  Noise norm50 shape: {norm50_map.shape}")

    # Apply bandpass to norm50 (highpass at 16 Hz like C++)
    norm50_map = _bandpass_rms(norm50_map, 16.0, f_high, sample_rate)

    # Apply whitening using C++ algorithm:
    # For each TF frequency bin m, interpolate nRMS across time and divide
    # This is equivalent to the per-pixel linear interpolation in C++

    # norm50_map shape: (M, K+1) where K+1 is number of measurements
    M = norm50_map.shape[0]
    n_time_tf = tf_map.data.shape[1]  # Number of time bins in TF domain
    K = norm50_map.shape[1] - 1

    # Create interpolated norm50 at full TF resolution
    # For each frequency bin, interpolate K+1 measurements to n_time_tf samples
    norm50_interp = np.zeros((M, n_time_tf), dtype=np.float64)
    for m in range(M):
        # Create interpolation function from the K+1 norm50 measurements
        # These measurements are uniformly spaced in time (stride-based)
        norm50_measurements = np.abs(norm50_map[m, :])  # K+1 values

        # Time indices of measurements (in TF samples, uniformly spaced)
        if K == 0:
            # Single measurement - use constant for all time
            norm50_interp[m, :] = norm50_measurements[0]
        else:
            # K+1 measurements uniformly spaced across time
            meas_indices = np.linspace(0, n_time_tf - 1, K + 1)
            tf_indices = np.arange(n_time_tf)
            norm50_interp[m, :] = np.interp(tf_indices, meas_indices, norm50_measurements)

    # Similarly interpolate median
    median_interp = np.zeros((M, n_time_tf), dtype=np.float64)
    for m in range(M):
        median_measurements = np.abs(median_map[m, :])  # K+1 values
        K = len(median_measurements) - 1

        if K == 0:
            median_interp[m, :] = median_measurements[0]
        else:
            meas_indices = np.linspace(0, n_time_tf - 1, K + 1)
            tf_indices = np.arange(n_time_tf)
            median_interp[m, :] = np.interp(tf_indices, meas_indices, median_measurements)

    # Whiten: (coeff - median) / norm50
    whitened_data = (tf_map.data - median_interp) / (norm50_interp + 1.0e-12)

    # Modify TF map data in place for whitening
    tf_map.data = whitened_data

    # Transform whitened coefficients back to time domain
    whitened_ts_gwpy = wdm.w2t(tf_map)

    # Convert gwpy TimeSeries back to pycbc
    whitened_data = np.array(whitened_ts_gwpy.value, dtype=np.float64)
    whitened_pycbc = pycbc.types.TimeSeries(
        whitened_data,
        delta_t=h_ts.delta_t,
        epoch=h_ts.start_time
    )

    # For nRMS, return the decimated version (not full-length interpolated)
    # The C++ code returns nRMS at decimated WDM resolution (~59,450 samples for 2.4M input)
    # Take the mean across frequency bins and inverse transform
    nrms_mean_freq = np.abs(norm50_interp).mean(axis=0)  # Average across M frequency bins, shape (n_time_tf,)

    # Create a wavearray from this 1D mean and get its original time resolution
    # The nRMS should correspond to the decimated time bins from WDM
    # tf_map.data has shape (M, n_time_tf). The n_time_tf corresponds to ~59,450 samples.
    nrms_pycbc = pycbc.types.TimeSeries(
        nrms_mean_freq,
        delta_t=1.0 / (sample_rate / (len(nrms_mean_freq) / len(signal_data))),  # Scaled delta_t for decimated
        epoch=h_ts.start_time
    )

    logger.info(f"  Whitened data: {len(whitened_pycbc)} samples")
    logger.info(f"  nRMS: {len(nrms_pycbc)} samples")

    return whitened_pycbc, nrms_pycbc


def _estimate_noise_rms(tf_map, edge_length=1.0, block_size=4096):
    """
    Estimate noise RMS in TF domain using block-median approach.

    Parameters
    ----------
    tf_map : TimeFrequencyMap
        TF map with `data` attribute (from wdm_wavelet).
    edge_length : float
        Edge length (seconds) excluded from RMS calculation.
    block_size : int
        Time-domain block size for local RMS estimation.

    Returns
    -------
    np.ndarray
        TF-structured noise RMS estimate (same shape as tf_map.data).
    """
    coeff = np.asarray(tf_map.data)
    if coeff.ndim != 2:
        raise ValueError("_estimate_noise_rms expects 2D TF map")

    _, T = coeff.shape
    power = np.abs(coeff) ** 2
    dt = float(tf_map.dt)

    # Exclude edge bins
    edge_bins = int(max(0, edge_length / dt))
    valid_start = edge_bins
    valid_stop = T - edge_bins

    if valid_stop <= valid_start:
        # If edge is too large, return uniform RMS
        return np.ones_like(power) * np.median(np.sqrt(power))

    nrms = np.zeros_like(power)

    # Per-frequency-bin RMS using median filter
    for m in range(coeff.shape[0]):
        freq_power = power[m, valid_start:valid_stop]
        if freq_power.size > 0:
            median_val = np.median(freq_power)
            nrms[m, :] = np.sqrt(np.maximum(median_val, 1.0e-12))
        else:
            nrms[m, :] = 1.0

    return nrms


def _estimate_noise_rms_cwb(tf_coeff, original_length, window_length=60.0, stride=60.0, edge_length=10.0, sample_rate=2048.0):
    """
    Estimate noise RMS using the cWB algorithm (matching C++ WSeries::white()).

    This replicates the C++ algorithm:
    1. Divide data into K intervals (based on window and stride) - using ORIGINAL audio duration
    2. For each interval, sort absolute values and compute:
       - Median (50% percentile)
       - 31% percentile bounds (15.865% and 84.135%)
       - norm50 = (upper_percentile - lower_percentile) / 2
    3. Linear interpolation in TF domain using K+1 measurements
    4. Apply to whiten: (coeff - median(t)) / norm50(t)

    Parameters
    ----------
    tf_coeff : TimeFrequencyMap
        WDM coefficient object with `.data` attribute (complex TF map).
    original_length : int
        Length of ORIGINAL audio signal (before TF transform).
    window_length : float
        Window length (seconds) for noise measurement.
    stride : float
        Stride (seconds) between measurements.
    edge_length : float
        Edge length (seconds) excluded from computation.
    sample_rate : float
        Sample rate (Hz).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (median_map, norm50_map) - both with shape (M, n_time_tf) matching tf_coeff.data
    """
    coeff = np.asarray(tf_coeff.data)
    if coeff.ndim != 2:
        raise ValueError("Expected 2D TF coefficient array")

    M, n_time_tf = coeff.shape  # M is freq bins, n_time_tf is time samples in TF domain

    # Use ORIGINAL audio duration for noise measurement interval calculations
    orig_offset_samples = int(edge_length * sample_rate + 0.5)
    if orig_offset_samples & 1:  # Make offset even (like C++)
        orig_offset_samples -= 1

    # Duration in ORIGINAL audio domain
    orig_duration = original_length / sample_rate

    # Compute window size and number of measurements from ORIGINAL domain
    if window_length <= 0:
        window_length = orig_duration - 2.0 * edge_length

    if stride > window_length or stride <= 0:
        stride = window_length

    K = int((orig_duration - 2.0 * edge_length) / stride)
    if K < 1:
        K = 1

    logger.debug(f"    RMS est: orig_length={original_length}, orig_duration={orig_duration:.3f}s, window_length={window_length:.3f}s, stride={stride:.3f}s -> K={K}")

    # Use original audio samples
    n_usable = original_length - 2 * orig_offset_samples
    k_samples = n_usable // K
    if k_samples & 1:  # Make k even
        k_samples -= 1

    window_samples = int(window_length * sample_rate + 0.5)

    # Indices for percentiles (matching C++ logic)
    mm = window_samples // 2  # Median index
    mL = int(0.15865 * window_samples + 0.5)  # -1 sigma (~16%)
    mR = window_samples - mL - 1  # +1 sigma (~84%)

    # Calculate starting positions
    jL = (n_usable - k_samples * K) // 2
    jj_start = jL - mm

    logger.debug(f"    _estimate_noise_rms_cwb: K={K}, window_samples={window_samples}, mm={mm}, mL={mL}, mR={mR}")
    logger.debug(f"    jL={jL}, jj_start={jj_start}, k_samples={k_samples}")

    # Initialize output arrays (K+1 measurements per frequency bin)
    median_vals = np.zeros((M, K + 1))
    norm50_vals = np.zeros((M, K + 1))

    # IMPORTANT: We need to read from ORIGINAL audio domain samples, but we only have TF coefficients!
    # We need to map TF samples back to original audio samples.
    # The TF coefficient at time index t_tf corresponds to approximate audio sample t_orig where:
    # t_orig ≈ t_tf * (original_length / n_time_tf)

    # Compute power for all coefficients (in TF domain)
    power = np.abs(coeff)

    # Mapping from original audio sample index to TF sample index
    orig_to_tf_scale = n_time_tf / original_length

    # Process each frequency bin
    for m in range(M):
        freq_power = power[m, :]

        for j in range(K + 1):
            # Determine window position IN ORIGINAL AUDIO DOMAIN
            jj = jj_start + j * k_samples  # Sample index in original audio

            # Handle boundaries in original domain
            if jj < orig_offset_samples:
                p_start_orig = orig_offset_samples
            elif jj + window_samples > original_length - orig_offset_samples:
                p_start_orig = original_length - orig_offset_samples - window_samples
            else:
                p_start_orig = jj

            p_end_orig = p_start_orig + window_samples
            if p_end_orig > original_length:
                p_end_orig = original_length
                p_start_orig = max(0, p_end_orig - window_samples)

            # Convert to TF domain indices
            p_start_tf = max(0, int(np.floor(p_start_orig * orig_to_tf_scale)))
            p_end_tf = min(n_time_tf, int(np.ceil(p_end_orig * orig_to_tf_scale)))

            # Extract window and sort
            window_data = freq_power[p_start_tf:p_end_tf]
            if len(window_data) < 3:
                median_vals[m, j] = 1.0
                norm50_vals[m, j] = 1.0
                continue

            sorted_data = np.sort(window_data)

            # Compute median and percentile bounds
            median = sorted_data[min(mm, len(sorted_data) - 1)]
            lower = sorted_data[min(mL, len(sorted_data) - 1)]
            upper = sorted_data[min(mR, len(sorted_data) - 1)]

            median_vals[m, j] = median
            norm50_vals[m, j] = (upper - lower) / 2.0

    # Interpolate to full TF resolution (matching C++ linear interpolation)
    median_map = np.zeros((M, n_time_tf))
    norm50_map = np.zeros((M, n_time_tf))

    # Measurements are at positions (in original audio samples): jL + j*k_samples for j=0..K
    # Convert to TF domain for interpolation
    meas_tf_indices = np.array([(jL + j * k_samples) * orig_to_tf_scale for j in range(K + 1)])

    for m in range(M):
        # Interpolate across all TF time bins
        tf_indices = np.arange(n_time_tf)
        median_map[m, :] = np.interp(tf_indices, meas_tf_indices, median_vals[m, :])
        norm50_map[m, :] = np.interp(tf_indices, meas_tf_indices, norm50_vals[m, :])

    # Ensure no zeros
    norm50_map = np.maximum(norm50_map, 1.0e-12)

    return median_map, norm50_map


def _bandpass_rms(nrms_map, f_low, f_high, sample_rate):
    """
    Apply high-pass and low-pass bandpass filter to RMS estimate.

    Parameters
    ----------
    nrms_map : np.ndarray
        2D noise RMS map.
    f_low : float
        High-pass cutoff (Hz).
    f_high : float
        Low-pass cutoff (Hz).
    sample_rate : float
        Sample rate (Hz).

    Returns
    -------
    np.ndarray
        Bandpass-filtered RMS map.
    """
    if nrms_map.ndim != 2:
        raise ValueError("Expected 2D RMS map")

    M, _ = nrms_map.shape
    output = np.copy(nrms_map)

    # Design Butterworth filters
    nyquist = sample_rate / 2.0
    f_high_norm = min(float(f_high) / nyquist, 0.99)
    f_low_norm = min(float(f_low) / nyquist, 0.99)

    # High-pass (remove low frequencies)
    if f_low > 0:
        try:
            b_hp, a_hp = scipy_signal.butter(2, f_low_norm, btype='high')
            for m in range(M):
                output[m, :] = scipy_signal.filtfilt(b_hp, a_hp, output[m, :])
        except Exception as e:
            logger.warning(f"High-pass filter failed: {e}")

    # Low-pass (remove high frequencies if f_high < Nyquist)
    if f_high < nyquist * 0.99:
        try:
            b_lp, a_lp = scipy_signal.butter(2, f_high_norm, btype='low')
            for m in range(M):
                output[m, :] = scipy_signal.filtfilt(b_lp, a_lp, output[m, :])
        except Exception as e:
            logger.warning(f"Low-pass filter failed: {e}")

    return np.maximum(output, 1.0e-12)


def _whiten_coefficients(tf_coeff, nrms_map, regularization=1.0e-6):
    """
    Whiten TF coefficients by division by noise RMS.

    Parameters
    ----------
    tf_coeff : object
        WDM coefficient object with `.data` (complex).
    nrms_map : np.ndarray
        Noise RMS estimate (2D, same shape as tf_coeff.data).
    regularization : float
        Small value to avoid division by zero.

    Returns
    -------
    object
        Whitened coefficient object (with modified `.data`).
    """
    coeff = np.asarray(tf_coeff.data, dtype=np.complex128)
    nrms_safe = np.maximum(nrms_map, regularization)

    # Divide each coefficient by RMS
    whitened = coeff / (nrms_safe + 1.0e-12)

    # Create output object (reuse structure)
    tf_coeff.data = whitened
    return tf_coeff


def _apply_wiener_filter(tf_coeff, nrms_map, snr_threshold=1.0):
    """
    Apply Wiener filter to TF coefficients.

    Wiener weight = signal_power / (signal_power + noise_power)
    Simplified: weight = |coeff|^2 / (|coeff|^2 + noise_power)

    Parameters
    ----------
    tf_coeff : object
        WDM coefficient object with `.data` (complex).
    nrms_map : np.ndarray
        Noise RMS estimate (2D).
    snr_threshold : float
        Threshold for interpreting coefficients as signal.

    Returns
    -------
    object
        Filtered coefficient object.
    """
    coeff = np.asarray(tf_coeff.data, dtype=np.complex128)
    power = np.abs(coeff) ** 2
    noise_power = nrms_map ** 2

    # Wiener weights
    signal_plus_noise = power + noise_power + 1.0e-12
    weights = power / signal_plus_noise

    # Apply weights
    filtered = coeff * weights

    # Create output object
    tf_coeff.data = filtered
    return tf_coeff


def _average_phases(tf_coeff):
    """
    Average 0-phase and 90-phase (I/Q) representations.

    In WDM, phases are interleaved in the time blocks:
    [0-phase: coeff_0_0, coeff_0_1, ..., coeff_0_T/2,
     90-phase: coeff_90_0, coeff_90_1, ..., coeff_90_T/2]

    This function averages them back together.

    Parameters
    ----------
    tf_coeff : object
        WDM coefficient object with `.data` (complex).

    Returns
    -------
    object
        Averaged coefficient object.
    """
    coeff = np.asarray(tf_coeff.data, dtype=np.complex128)

    if coeff.ndim != 2:
        return tf_coeff

    _, T = coeff.shape

    # Assume even T for phase interleaving
    if T % 2 != 0:
        logger.warning("Odd time length; averaging both phases may produce unexpected results")
        return tf_coeff

    T_half = T // 2

    # Extract phases
    phase_0 = coeff[:, :T_half]
    phase_90 = coeff[:, T_half:]

    # Average
    averaged = (phase_0 + phase_90) / 2.0

    # Reconstruct by concatenating both copies
    result = np.hstack([averaged, averaged])
    coeff_out = tf_coeff
    coeff_out.data = result

    return coeff_out
