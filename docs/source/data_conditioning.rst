.. _data_conditioning:

Data Conditioning
=================

.. stage-nav:: search
   :current: conditioning

This guide explains what pycWB does to each detector's strain inside a job
segment before the time-frequency analysis: resampling, line regression,
whitening, and the noise RMS (nRMS) estimate that later stages reuse.

.. contents:: Table of Contents
   :depth: 2
   :local:


Why this matters
----------------

Pixel selection sums pixel energies over detectors and applies one threshold
at every frequency of a WDM resolution, so all data need a common noise scale.
Whitening provides it, after regression has removed narrow spectral lines. The
nRMS map is reused later: the likelihood weights detectors with it, and
reconstruction uses it to return whitened amplitudes to strain. If one band
dominates the background, or ``hrss`` values look wrong, check this stage.


How it works
------------

Conditioning runs once per job segment and injection trial, before the lag
loop; every lag reuses its output. The input is each detector's strain at
``inRate`` over the padded job window, with injections added
(:doc:`data_ingestion`). The native segment processor then runs, in order:

1. **Resampling** to the analysis rate ``rateANA``.
2. **Regression** of predictable narrow-band components, for all detectors
   before any whitening.
3. **Whitening**, which also produces the nRMS map.
4. **Conditioning hooks**, if configured.
5. **Injection whitening**: signal-only injection strains are whitened with the
   same nRMS map, for comparison with the reconstruction.

The cWB-2G products are ``HoT`` (whitened strain) and ``nRMS``; pycWB returns
a conditioned ``TimeSeries`` and a ``NoiseRMSMap`` per detector.

Resampling
~~~~~~~~~~

:py:func:`~pycwb.modules.read_data.data_check.check_and_resample_py` rejects
data with NaNs or not at ``inRate``, and multiplies by ``dcCal[i]`` if that is
positive and not 1. It resamples to ``fResample`` if positive, then by a
further :math:`2^{\text{levelR}}`, and multiplies the samples by
:math:`2^{\text{levelR}/2}` (the Meyer-downsampling gain, :math:`\sqrt{2}` per
level).

.. math::

   \text{rateANA} = \frac{f_\text{in}}{2^{\text{levelR}}}, \qquad
   f_\text{in} = \begin{cases}
     \text{fResample} & \text{if fResample} > 0 \\
     \text{inRate}    & \text{otherwise}
   \end{cases}

The defaults give 16384 / 2\ :sup:`2` = 4096 Hz. Both steps use the FFT
resampler :py:meth:`~pycwb.types.time_series.TimeSeries.cwb_resampling` (cWB
``wavearray::Resample``), which needs an even whole number of output samples.

With ``injection_resampling: cwb`` (default), ``target_snr`` injections are
resampled separately by
:py:func:`~pycwb.modules.data_conditioning.resampling.resample_snr_injection`
(FFT to ``fResample`` if set, then Meyer(1024) downsampling) and added
afterwards, so ``dcCal`` does not scale them. In this mode, trials mixing
target-SNR and fixed-``hrss`` injections are rejected. See
:doc:`tutorial_injection`.

Regression
~~~~~~~~~~

:py:func:`~pycwb.modules.data_conditioning.regression.apply_regression` follows
cWB's LPE (linear-prediction error) ``regression``, with the strain as its own
witness:

1. Transform the strain with a WDM of ``rateANA / 8`` layers (4 Hz
   resolution). A mean-subtracted copy is transformed as the witness.
2. For each layer from 1 Hz to ``fHigh``, correlate at lags
   :math:`-K \dots K` (:math:`K` = ``REGRESSION_FILTER_LENGTH``) in both
   quadratures. Lag 0 is excluded, so each coefficient is predicted from its
   neighbours. Averages skip ``segEdge`` at both ends and keep only the
   ``REGRESSION_MATRIX_FRACTION`` of samples with the smallest magnitude.
3. Solve by eigen-decomposition, keeping at most ``REGRESSION_SOLVE_EIGEN_NUM``
   components above ``REGRESSION_SOLVE_EIGEN_THR``.
4. Subtract a layer's prediction only if its RMS, relative to the layer's
   trimmed RMS, is at least ``REGRESSION_APPLY_THR``. The subtraction is done
   in the time domain, averaging the 00 and 90 phase inversions.

Persistent lines are predictable and are removed; broadband noise is not.
Nothing is subtracted if ``REGRESSION_FILTER_LENGTH`` ≤ 0, the segment is too
short, or no layer passes the threshold.

Whitening
~~~~~~~~~

``whiteMethod`` is ``wavelet`` (default; alias ``python``) or ``mesa``.

**Wavelet**
(:py:func:`~pycwb.modules.data_conditioning.whitening.whiten_wavelet`, cWB
``WSeries::white`` mode 0) uses a WDM of :math:`2^{\text{l\_white}}` layers, or
:math:`2^{\text{l\_high}}` if ``l_white`` is 0 (256 layers at 8 Hz by
default). Pixel power is :math:`P_f(t) = a_{00}^2 + a_{90}^2`. A segment of
duration :math:`T` (margins included) is split into
:math:`K = \lfloor (T - 2\,\text{segEdge}) / \text{whiteStride} \rfloor` equal
intervals, giving :math:`K+1` noise anchors. At each anchor :math:`\tau_j`:

.. math::

   \sigma_f(\tau_j) =
   \sqrt{0.7191 \cdot \operatorname{median}_{t \in W_j} P_f(t)}

:math:`W_j` is ``whiteWindow`` long, centred on the anchor and kept
``segEdge`` away from the ends. For Gaussian noise the median of :math:`P` is
:math:`2\ln 2\,\sigma^2`, and 0.7191 is close to :math:`1/(2\ln 2)`, so
:math:`\sigma_f` is the per-quadrature RMS. ``whiteWindow: 0`` uses the whole
span. If ``whiteStride`` is not positive or exceeds ``whiteWindow``, it is set
to ``whiteWindow``. Rows outside (16 Hz, ``fHigh``) get :math:`\sigma_f = 1`
(cWB ``bandpass(16, 0, 1)``) and are not rescaled; this 16 Hz edge ignores
``fLow``. Coefficients are divided by :math:`\sigma_f`, interpolated between
anchors, and transformed back as the mean of the 00 and 90 inversions. The
defaults give 31 anchors per 600 s job, each from 60 s of data.

**MESA**
(:py:func:`~pycwb.modules.data_conditioning.whitening_mesa.whiten_mesa`,
optional ``memspectrum`` and ``scikit-learn``) removes the mean and applies an
8th-order zero-phase Butterworth high-pass at ``fLow``. It fits maximum-entropy
PSDs (``mesaOrder``, ``mesaSolver``) to ``mesaWindow``-second windows every
``mesaStride`` = ``mesaWindow / 3`` seconds (enforced). If ``mesaHalfSeg`` > 0,
a running median over 2 × ``mesaHalfSeg`` + 1 estimates smooths the PSDs. If
``mesaReindex`` is set, PSDs flagged by an IsolationForest are replaced by the
nearest inlier. Each window is Planck-tapered, divided by
:math:`\sqrt{\text{PSD}}` in the Fourier domain, and stitched by thirds. The
anchors come from the ratio of high-passed to whitened WDM magnitudes, on the
same lattice and band.

The noise RMS map
~~~~~~~~~~~~~~~~~

A :py:class:`~pycwb.types.noise_rms.NoiseRMSMap` holds only the anchors,
shape ``(n_frequency, K+1)``, timed by ``noise_start`` and ``noise_rate``.
After pixel selection,
:py:func:`~pycwb.modules.data_conditioning.noise.lookup_pixel_noise_rms` (cWB
``detector::setrms``) takes, for each pixel, the anchor interval containing
its lag-shifted time (no interpolation) and the :math:`N_f` rows covering its
band:

.. math::

   \sigma_\text{pix} = \sqrt{N_f \Big/ \textstyle\sum_f \sigma_f^{-2}}

The likelihood weights each detector by :math:`1/\sigma_\text{pix}`,
normalised over the network. Reconstruction multiplies whitened amplitudes by
:math:`\sigma_\text{pix}` to return to strain units.

Conditioning hooks
~~~~~~~~~~~~~~~~~~

:py:func:`~pycwb.modules.conditioning_plugins.api.run_hooks` runs the modules
under ``conditioning.post_whitening``, then ``selection.time_vetoes``, checking
options against each ``OPTIONS_SCHEMA``. A hook may not change the detector
count or the strain timeline. Bundled plugins:

- :py:mod:`~pycwb.modules.conditioning_plugins.o3a_conditioning`
  (``post_whitening``): O3a 16–48 Hz correction (H1 and L1 by default); it
  attaches a noise-variation map (cWB ``nVAR``) to the nRMS.
- :py:mod:`~pycwb.modules.conditioning_plugins.cwb_gating` (``time_vetoes``):
  excludes padded times of high whitened-strain energy; the strain is kept.

Excluded intervals are removed from the CAT2 keep windows (or the whole
analysis span), so no pixels are selected there. The recorded livetime excludes
them, but the ``segTHR`` check uses the livetime before hooks. Hook runs are
recorded in ``conditioning/job_<index>/trial_<trial>/diagnostics.json``
(:doc:`tutorial_conditioning`).


Implementation
--------------

- :py:mod:`pycwb.modules.data_conditioning` entry points: ``condition_strains``
  (all detectors) and ``condition_strain`` (one detector, called per detector
  by the online search and by the CUDA workflow's ``gpu.condition_workers``).
- Regression kernels: ``regression_numba`` (default) or ``regression_jax``,
  set by ``execution_profile.regression_engine`` (``numba``); JAX is also used
  if Numba cannot be imported.
- :py:mod:`pycwb.modules.data_conditioning_root` is the legacy ROOT-backed
  variant, which calls cWB's C++ regression and whitening.


Configuration
-------------

Defaults are from :py:mod:`pycwb.constants.user_parameters_schema` (full
list: :ref:`schema`).

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - Parameter
     - Default
     - Meaning
   * - ``inRate`` / ``fResample`` / ``levelR``
     - 16384 / 0 / 2
     - Input rate [Hz] / intermediate rate (0 = none) / downsampling level
   * - ``dcCal`` / ``segEdge``
     - ``[1.0, …]`` / 8.0
     - Per-detector amplitude factor / margin [s] kept out of noise statistics
   * - ``fLow`` / ``fHigh``
     - 64.0 / 2048.0
     - MESA high-pass corner / top of regression layers and nRMS band [Hz]
   * - ``REGRESSION_FILTER_LENGTH`` / ``_MATRIX_FRACTION`` / ``_APPLY_THR``
     - 8 / 0.95 / 0.8
     - Filter half-length :math:`K` (≤ 0 disables) / trimmed-average
       fraction / minimum relative prediction RMS to subtract a layer
   * - ``REGRESSION_SOLVE_EIGEN_THR`` / ``_NUM`` / ``_REGULATOR``
     - 0.0 / 10 / ``h``
     - Eigenvalue threshold (< 0: relative) / maximum components / ``h`` drops
       the rest, ``s`` and ``m`` use the last-kept and largest eigenvalue
   * - ``whiteMethod`` / ``l_white``
     - ``wavelet`` / 0
     - ``wavelet`` (alias ``python``) or ``mesa`` / WDM level (0: ``l_high``)
   * - ``whiteWindow`` / ``whiteStride``
     - 60.0 / 20.0
     - Anchor window (0 = whole span) / nominal anchor spacing [s]
   * - ``mesaWindow`` / ``mesaStride`` / ``mesaOrder``
     - 15 / 5 / 800
     - MESA PSD window / stride [s] (forced to window / 3) / maximum AR order
   * - ``mesaSolver`` / ``mesaHalfSeg`` / ``mesaReindex``
     - ``Fast`` / 4 / true
     - Levinson solver (``Fast``, ``Standard``) / PSD running-median
       half-width (< 1 = off) / IsolationForest outlier replacement
   * - ``injection_resampling``
     - ``cwb``
     - ``cwb``: Meyer path for target-SNR injections; ``fft``: FFT only
   * - ``conditioning.post_whitening``, ``selection.time_vetoes``
     - none
     - Hook lists of ``module`` and ``options`` entries


Output
------

For each detector, the WDM stage receives the **conditioned strain** (cWB
``HoT``), a ``TimeSeries`` at ``rateANA`` with the same start and length as the
padded input, and the **NoiseRMSMap** (with any variation map). Injection
trials add whitened injection strains (rows from ``fLow`` to ``fHigh`` only)
and unwhitened WDM round-trip copies. Coherence setup transforms the
conditioned strains at each level from ``l_low`` to ``l_high``
(:doc:`wdm_transform`).


Validation checks
-----------------

- **Analysis rate**: the ``Resampling data from … to …`` log lines should
  end at ``rateANA``. ``Sample rate is not consistent`` means the input is
  not at ``inRate``.
- **Whitened level**: in stationary Gaussian noise, in-band whitened WDM
  coefficients have roughly unit RMS per quadrature. Residual lines or a
  sloped spectrum point to regression or ``whiteWindow`` problems.
- **Regression effect**: compare with a ``REGRESSION_FILTER_LENGTH: 0`` run
  on the same data. Lines should shrink while the broadband level stays the
  same. Also compare background and recovered injections.
- **MESA reproducibility**: ``mesaReindex`` uses an unseeded IsolationForest,
  so results are not bit-for-bit repeatable. Disable it for exact comparisons.


----

**See also:** :doc:`pipeline_lifecycle` · :doc:`data_ingestion` · :doc:`tutorial_conditioning`

**Next:** :doc:`wdm_transform` — the multi-resolution time-frequency transform
