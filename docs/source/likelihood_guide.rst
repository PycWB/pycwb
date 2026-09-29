.. _likelihood_guide:

Likelihood
==========

.. stage-nav:: search
   :current: likelihood

This guide describes pycWB's likelihood framework—the mathematical core that
assigns a ranking statistic to each candidate event. It covers the Dominant
Polarization Frame, sky scan, SNR definitions, regularization, and the
configurable parameters that control likelihood evaluation.

.. contents:: Table of Contents
   :depth: 2
   :local:


Why this matters
----------------

The likelihood is the mathematical core that separates signals from noise.
Most users only need to set ``netRHO``, ``netCC``, and ``healpix``. The
other parameters (:math:`\delta`, ``cfg_gamma``, ``precision``) are for
expert tuning and can degrade performance if set incorrectly.


Overview
--------

The pycWB likelihood pipeline follows the **cWB 2G likelihood WP algorithm**.
For each supercluster (candidate event), the algorithm:

1. Projects the multi-detector time-frequency data onto the **Dominant
   Polarization Frame (DPF)** for each sky direction.
2. Scans the sky to find the best-fit direction.
3. Computes coherent statistics: SNR (:math:`\rho`), network correlation
   (:math:`cc`), null energy, and :math:`\chi^2`.
4. Applies regularization to handle degenerate configurations (e.g.,
   2-detector networks).
5. Populates per-pixel detection statistics and extracts waveform
   reconstructions.

The production code lives in :py:mod:`pycwb.modules.likelihoodWP`, with the
sky scan in :py:mod:`~pycwb.modules.likelihoodWP.sky_scan` and statistics
computation in :py:mod:`~pycwb.modules.likelihoodWP.sky_statistics`.

In the cWB-2G stage flow, likelihood reads each surviving supercluster, loads
time-delay amplitudes for its pixels, evaluates ``likelihood2G`` or
``likelihoodWP``, reconstructs event parameters, and optionally produces a
Coherent Event Display. pycWB performs the same algorithmic role but writes
structured trigger data, reconstructed waveforms, plots, and postproduction
inputs through the Python workflow.


Dominant Polarization Frame (DPF)
---------------------------------

The DPF formalism determines the optimal polarization basis for each sky
direction. For a given sky direction, the antenna response functions
:math:`F_+` and :math:`F_\times` determine how each detector responds to the
two gravitational-wave polarizations.

**Noise-weighted antenna patterns** (per sky direction, per pixel, per
detector :math:`i`):

.. math::

   f_i = w_i \, F_{+,i}, \qquad F_i = w_i \, F_{\times,i}, \qquad
   w_i = \frac{\sigma_i^{-1}}{\sqrt{\sum_k \sigma_k^{-2}}}

where :math:`\sigma_i` is the noise RMS of detector :math:`i` at the pixel's
time-frequency location (from the conditioning ``nRMS`` maps), so that
:math:`\sum_i w_i^2 = 1`.

**Optimal polarization angle** :math:`\psi`:

.. math::

   ff &= f \cdot f, \quad FF = F \cdot F, \quad fF = F \cdot f \\[4pt]
   \sin 2\psi &= \frac{2 fF}{\sqrt{(ff - FF)^2 + (2 fF)^2}} \\[4pt]
   \cos 2\psi &= \frac{ff - FF}{\sqrt{(ff - FF)^2 + (2 fF)^2}}

**Effective plus-polarization response**:

.. math::

   |f_+|^2 = \frac{ff + FF + \sqrt{(ff - FF)^2 + (2fF)^2}}{2}

After the rotation by :math:`\psi`, the cross vector is made orthogonal to the
plus vector, :math:`F \leftarrow F - f\,(f\cdot F)/|f_+|^2`, and
:math:`|f_\times|^2 = F\cdot F`. In the DPF the :math:`+` response is maximal
and the :math:`\times` response is orthogonal to it. Both components enter the
signal reconstruction; the :math:`\times` component is regulated (see
``REG[1]`` below and :ref:`the sky-scan projection <likelihood_projection>`).

**Network index.** Per pixel :math:`\nu = \sum_k f_k^4 / |f_+|^4`, and per sky
direction

.. math::

   NI = \sqrt{\frac{1}{N_{pix}} \sum_{pix} \frac{|f_\times|^2}{\nu}}

**DPF regulator** ``REG[1]``
(:py:func:`pycwb.modules.likelihoodWP.dpf.compute_dpf_regulator`). The
threshold is

.. math::

   \gamma_{reg} = \frac{2}{3}\,\gamma^2

(:math:`\gamma` = ``cfg_gamma``; its sign does not enter here), and the
regulator is a single number per cluster:

.. math::

   \mathrm{REG}[1] = \left(\frac{N_{sky}^2}{n_{valid}^2} - 1\right) \cdot E_{th},
   \qquad E_{th} = 2\,A_{core}^2\,n_{IFO}

where :math:`N_{sky}` is the number of unmasked sky directions and
:math:`n_{valid}` is the number of those with :math:`NI > \gamma_{reg}`.
``REG[1]`` grows when few directions have appreciable :math:`\times`
sensitivity, and it suppresses the :math:`\times` component of the
reconstructed signal in every direction. It does not act as a per-direction
sky penalty.


Sky Scan
--------

The sky scan (:py:func:`pycwb.modules.likelihoodWP.sky_scan.scan_sky`)
is the computational core of the likelihood pipeline. It visits every
unmasked HEALPix direction (see ``sky_mask``). Directions whose integer
detector delays are identical are grouped; the Numba ``prange`` runs over
these groups, and the delayed-data load is shared within a group. For each
direction:

1. **Apply time delay**: each pixel carries, per detector and per quadrature
   (0° and 90°), a vector of :math:`2K+1` time-delayed WDM amplitudes
   :math:`TD_i`. For sky direction :math:`l`, detector :math:`i` uses

   .. math::

      x_i = TD_i\left[K + ml[i, l]\right], \qquad
      ml[i, l] = \mathrm{rint}\left(\frac{(\vec R_i - \vec R_{ref})\cdot\hat n_l}{c}\,
      \mathrm{TDRate}\right) \in [-K, K]

   with :math:`\mathrm{TDRate} = \texttt{rateANA} \cdot \texttt{upTDF}` and
   :math:`K = \max(\texttt{TDSize}\cdot\texttt{upTDF},\ \lfloor \tau_{max}\,
   \mathrm{TDRate} \rfloor + 1)`, where :math:`\tau_{max}` is the maximum
   network light-travel delay. Delays are relative to ``refIFO``.

2. **Select and project onto the DPF**
   (:py:func:`pycwb.modules.likelihoodWP.sky_kernels.project_signal_packet`).
   Only pixels with network energy
   :math:`E_p = \sum_i (x_{i,0}^2 + x_{i,90}^2) > E_{th}` are used.

   .. _likelihood_projection:

   With :math:`a_+ = |x\cdot f|^2` and :math:`a_\times = |x\cdot F|^2`
   (summed over both quadratures), the regulated amplitudes are

   .. math::

      u &= \frac{x\cdot f}{\max\left(|f_+|^2,\ \mathrm{REG}[0]\sqrt{\nu\,a_+/E_p}\right)} \\[4pt]
      v &= \frac{x\cdot F}{\max\left(|f_\times|^2,\ R\,\sqrt{a_\times}/|u|\right)},
      \qquad R = 0.1 + \frac{\mathrm{REG}[1]}{E_p}

   and the reconstructed detector response is :math:`s_i = f_i\,u + F_i\,v`.

3. **Orthogonalize** the 0° and 90° quadrature signal vectors of each pixel
   (a per-pixel rotation between the two quadratures).

4. **Compute coherent statistics**. Per pixel: signal energy
   :math:`\ell = |s|^2`, residual :math:`r = |s - x|^2`, a Gaussian-noise term
   :math:`g`, and the coherent energy (signal energy with the single-detector
   terms removed, per quadrature)

   .. math::

      e_c = |s|^2 \left(1 - \frac{\sum_i (s_i x_i)^2}{\left(\sum_i s_i x_i\right)^2}\right)

   Summing over pixels gives :math:`E_c = \sum e_c`, the null energy
   :math:`N = \tfrac12\sum (r + g)`, the data energy
   :math:`E = \tfrac12 \sum E_p`, and the likelihood :math:`L = E - N`. With
   :math:`M` the number of selected pixels:

   .. math::

      C_r &= \frac{2\sum \ell\, c}{\sum \ell}, \qquad c = \frac{e_c}{2|e_c| + r + g} \\[4pt]
      ch &= \frac{N}{n_{IFO} M + \sqrt{M}}, \qquad
      C_o = \frac{E_c}{E_c + N \max(ch, 1) - M (n_{IFO} - 1)}

   Directions with reduced correlation :math:`C_r <` ``netCC`` are skipped.

The selected direction :math:`l_{max}` maximizes the sky statistic
:math:`AA = L \cdot C_o` over the unmasked directions (on ties the last index
wins). A cluster whose maximum :math:`AA` is not positive is rejected.


SNR Definitions
---------------

At :math:`l_{max}`,
:py:func:`~pycwb.modules.likelihoodWP.sky_statistics.compute_statistics_at_sky_position`
rebuilds the data and signal packets with cross-talk corrections and computes:

- :math:`E_c`: core coherent energy — the sum of :math:`e_c` over core pixels,
  divided by :math:`\mathrm{norm} = \max\left(1, (E_o - E_h)/E_m\right)`, where
  :math:`E_o` is the total data energy, :math:`E_h` the satellite (halo)
  energy and :math:`E_m` the cross-talk-corrected packet data energy.
- :math:`R_c`: the fraction of coherent energy kept after the per-pixel
  Gaussian-noise correction,
  :math:`R_c = \sum e_c / \max(1, g/2) \,/\, \sum e_c`.
- :math:`\chi^2` (see below).

pycWB supports two SNR definitions, selectable via ``xgb_rho_mode``:

**cWB 2G SNR** (:math:`\rho`, ``xgb_rho_mode: false``):

.. math::

   \rho = \sqrt{\frac{E_c R_c}{2}}

The reported ``rho[0]`` is :math:`\rho / \sqrt{\max(\chi^2_w, 1)}`, using the
waveform-domain :math:`\chi^2_w` defined below.

**XGBoost** :math:`\rho_0` (``xgb_rho_mode: true``, the cWB XGB.rho0
convention):

.. math::

   \rho_0 = \sqrt{\frac{E_c}{1 + \chi^2\left(\max(1, \chi^2) - 1\right)}}

It is reported unchanged as ``rho[0]``; ``rho[1]`` then holds the 2G value
:math:`\sqrt{E_c R_c/2}\,/\sqrt{\max(\chi^2_w, 1)}`. Setting ``netRHO < 0`` is
the deprecated way to enable this mode.

**Selection cuts**
(:py:func:`~pycwb.modules.likelihoodWP.detection_statistics.get_likelihood_rejection_reason`).
A cluster is rejected if :math:`L_m \le 0`, :math:`E_o - E_h \le 0`,
:math:`N_{eff} < 1`, or

- 2G: :math:`E_c R_c / \max(\chi^2, 1) < 2\,\texttt{netRHO}^2`
- XGB: :math:`\rho_0 < |\texttt{netRHO}|`

where :math:`L_m = E_m - N_p - G_n` is the packet signal energy.

**Network Correlation** (:math:`cc`):

The fraction of coherent energy relative to coherent plus null energy:

.. math::

   \texttt{netcc[0]} = \frac{E_c R_c}{E_c R_c + D_c + N_w + G_n - N_{eff}(n_{IFO} - 1)}

where :math:`D_c` is the signal-packet coherent energy minus :math:`E_c`
(same normalization) and :math:`N_w` is the waveform-domain null energy.
``netcc[1]`` uses the same expression with :math:`(D_c + N_w + G_n)`
multiplied by :math:`1 + 2(\chi^2_w - 1)(1 - R_c)` when :math:`\chi^2_w > 1`.
The ``netCC`` threshold (default 0.5) is applied during the sky scan to
:math:`C_r`, not to the final ``netcc``.

**Chi-squared** (:math:`\chi^2`):

.. math::

   \chi^2 = \frac{N_p + G_n}{N_{eff} \cdot n_{IFO}}

where :math:`N_p` is the cross-talk-corrected null packet energy, :math:`G_n`
is the Gaussian-noise correction, :math:`N_{eff}` is the effective number of
pixels per detector (minus one), and :math:`n_{IFO}` is the number of
interferometers. This pixel-domain :math:`\chi^2` is used for :math:`\rho_0`
and for the selection cut. The waveform-domain
:math:`\chi^2_w = (N_w + G_n)/(N_{eff}\, n_{IFO})` is used for the reported
2G ``rho``. The reported ``penalty`` is :math:`(N_w + G_n)` divided by
:math:`n_{IFO} N_{eff}` (by :math:`n_{IFO}` times the number of core pixels
when ``pattern = 0``). :math:`\chi^2 \sim 1` for well-modeled signals and
:math:`\chi^2 > 1` for glitches or poorly modeled events.


Regularization
--------------

:math:`\delta` Regulator (Amplitude)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prevents degenerate sky locations in 2-detector networks where the
polarization angle is unconstrained:

.. math::

   \delta \in [-1, 1], \quad \text{default } 0.5

If :math:`\delta = 0`, it is stored as :math:`0.00001` to avoid a truly
degenerate regulator. The amplitude regularization term is:

.. math::

   \mathrm{REG}[0] = \min(|\delta|, 1) \cdot \sqrt{2}

It enters the denominator of the :math:`+` projection
(:ref:`see the sky scan <likelihood_projection>`); larger :math:`|\delta|`
regularizes more strongly, saturating at :math:`\sqrt 2` for
:math:`|\delta| \ge 1`. The sign of :math:`\delta` does not change
``REG[0]``: :math:`\delta < 0` only makes the sky-localization probability use
:math:`L` instead of :math:`L \cdot C_o`; the selected direction is unchanged.

``cfg_gamma`` Regulator (Sky Location):

Regulates the :math:`\times` component through ``REG[1]`` (see the DPF
regulator above), via the threshold :math:`\gamma_{reg} = \tfrac23\gamma^2`.
:math:`\gamma = 0` effectively disables it (``REG[1]`` :math:`\approx 0` when
every direction has :math:`NI > 0`). :math:`\gamma < 0` additionally applies
an antenna-pattern prior to the sky-localization probability. Range
:math:`[-1, 1]`, default 0.5.


Detection Statistics
--------------------

After the selection cuts, detection statistics are populated per pixel and per
cluster via
:py:func:`pycwb.modules.likelihoodWP.detection_statistics.populate_detection_statistics`:

- **Core flags**: pixels whose network energy exceeds
  :math:`E_{th} = 2 A_{core}^2 n_{IFO}` and that are accepted by the
  projection at :math:`l_{max}`
- **Per-pixel arrays**: :math:`+` and :math:`\times` quadrature energies,
  cross-talk-corrected likelihood and null, and the data and signal packet
  amplitudes
- **Subnetwork statistic**: with :math:`S_i` the per-detector signal SNR,

  .. math::

     E_0 = \sum_i S_i - \max_i S_i, \qquad
     E_{sub} = E_0 \left(1 + 2 R_c \frac{E_0}{\max_i S_i}\right)

  and the stored statistic is

  .. math::

     \texttt{netcc[2]} = \frac{E_{sub}}{E_{sub} + N_{max}}, \qquad
     N_{max} = G_n + N_p - N_{eff}(n_{IFO} - 1)

- **Multi-resolution analysis (MRA) waveform**: whitened and strain-unit
  detector waveforms are synthesized from the packets to obtain the
  waveform-domain signal, data and null energies per detector, the
  time/frequency centroids, and the physical signal energy used for
  :math:`h_{rss}`

The MRA/XTalk catalog used here is not the WDM transform itself. WDM defines
the time-frequency basis; the catalog stores sparse overlaps between pixels at
different WDM resolutions and quadratures. The likelihood/reconstruction path
uses those overlaps to remove duplicated support between resolutions before
synthesizing the final detector waveforms.

**Chirp Mass**: the chirp mass

.. math::

   \mathcal{M} = \frac{(m_1 m_2)^{3/5}}{(m_1 + m_2)^{1/5}}

is estimated from the time-frequency track of the cluster pixels, but only
when all of the following hold: the ``execution_profile`` option
``native_chirp`` is enabled, ``xgb_rho_mode`` is true, ``optim`` is false,
``cfg_search`` is one of ``i e c r p b l s g``, and ``Search`` is ``CBC``,
``BBH`` or ``IMBHB``. The estimator is
:py:func:`pycwb.modules.likelihoodWP.chirp_micropixel.estimate_chirp`.
Otherwise the chirp-mass fields stay 0. In the default configuration the
Hough-transform fit
:py:func:`pycwb.modules.likelihoodWP.chirp_hough.update_chirp_mass_statistics`
runs for every accepted cluster regardless of ``Search``; it does not store a
chirp mass and only rescales
:math:`\texttt{rho[1]} = \texttt{rho[0]}\cdot\epsilon_{chirp}\sqrt{E_{frac}}`
in 2G mode with ``pattern`` :math:`\ne 0`.

**Sky localization**
(:py:func:`~pycwb.modules.likelihoodWP.detection_statistics.populate_sky_localization`):
the sky probability over directions with :math:`S_l > 0` is

.. math::

   p_l \propto \exp\left(-\frac{S_{max} - S_l}{2\sigma}\right)

where :math:`S` is the :math:`L \cdot C_o` sky map (:math:`L` when
:math:`\delta < 0`) and :math:`\sigma` is set by the waveform normalization,
:math:`R_c`, the number of core pixels and :math:`ch` at :math:`l_{max}`.
When ``cfg_gamma`` :math:`< 0`, :math:`p_l` is multiplied by
:math:`(A_l / A_{max})^4`, where :math:`A` is the energy-weighted antenna
sensitivity :math:`\sqrt{|f_+|^2 + |f_\times|^2}`. The square roots of the
10 %–90 % credible-region areas are stored as the error regions.


Big Cluster Optimization
------------------------

When a supercluster contains a large number of pixels, the full-resolution sky
scan becomes expensive. The ``precision`` parameter enables a coarse-grid
scan:

.. math::

   \text{csize} = |\text{precision}| \bmod 65536

If :math:`\text{csize} > 0`, :math:`n_{pixels} > nRES \times \text{csize}`
(counted after the ``BATCH`` pixel limit), and coarse-grid sky arrays were
passed to
:py:func:`~pycwb.modules.likelihoodWP.likelihood_setup.prepare_likelihood_inputs`
(``ml_big``, ``FP_big``, ``FX_big`` and ``big_cluster_healpix_order``), the
whole sky scan runs on that coarser HEALPix grid instead of the full grid.
There is no refinement pass. The native job-segment workflow does not
currently build these coarse arrays, so production runs always scan the
full-resolution grid. ``precision`` defaults to 0 (disabled).
``Config.get_precision(csize, order)`` returns
:math:`\text{csize} + 65536 \cdot \text{order}`, but likelihoodWP does not
decode the order part.


Usage Guidance
--------------

For standard users
~~~~~~~~~~~~~~~~~~

The three parameters you should set:

- ``netRHO``: coherent SNR threshold. Lower = more triggers, more background.
  Typical range: 3.5–5.0 for bursts.
- ``netCC``: network correlation threshold, applied to the reduced correlation
  :math:`C_r` of each sky direction during the scan. Lower = more triggers,
  more glitches. Typical: 0.4–0.6.
- ``healpix``: sky resolution. Higher = better localization but slower.
  Typical: 6–8.

Everything else should be left at defaults unless you have a specific reason
to change them.

For advanced users
~~~~~~~~~~~~~~~~~~

Tune these only with caution—they can degrade performance if set incorrectly:

- ``delta``: 2-detector sky regulator. :math:`|\delta|` sets ``REG[0]``;
  larger :math:`|\delta|` gives stronger regularization of the :math:`+`
  projection, saturating at :math:`|\delta| = 1`. A negative value also
  switches the sky-localization statistic from :math:`L \cdot C_o` to
  :math:`L`. Default 0.5.
- ``cfg_gamma``: sets :math:`\gamma_{reg} = \tfrac23\gamma^2` for ``REG[1]``,
  which suppresses the :math:`\times` component when few sky directions have
  network index above :math:`\gamma_{reg}`. A negative value also applies an
  antenna-pattern prior to the sky-localization probability.
- ``precision``: big-cluster coarse-grid scan
  (``csize = |precision| % 65536``). It only takes effect when coarse-grid sky
  arrays are supplied, which the native workflow does not currently do.
- ``xgb_rho_mode``: use :math:`\rho_0` (cWB XGB.rho0 convention) instead of
  the 2G :math:`\rho`; the SNR cut becomes :math:`\rho_0 \ge |\texttt{netRHO}|`.
  Set to ``true`` for XGBoost-ranked searches. It replaces the deprecated
  ``netRHO < 0`` convention.
- ``Search``: ``CBC`` / ``BBH`` / ``IMBHB`` is one of the conditions for the
  native chirp-mass estimate (together with ``xgb_rho_mode``,
  ``execution_profile.native_chirp``, ``optim: false`` and ``cfg_search``).

Developer notes
~~~~~~~~~~~~~~~

.. admonition:: Implementation detail
   :class: note

   The likelihood pipeline is the most computationally intensive part of pycWB.
   Key implementation details:

   - The sky scan is a Numba ``@njit(parallel=True)`` kernel. ``prange`` runs
     over groups of sky directions that share the same integer delay tuple;
     the ``execution_profile`` option ``sky_delay_reuse: false`` uses one
     direction per group.
   - DPF projection and coherent statistics are recomputed for every direction
     from the delay indices ``ml`` and the antenna patterns ``FP``/``FX``,
     which are built once per job segment.
   - JAX device buffers must be explicitly freed after each lag to prevent
     memory leaks—this is a known pitfall.
   - Big-cluster mode (``bBB``) replaces the full sky grid with a coarse grid
     for the whole scan; there is no refinement step, and it is inactive
     unless coarse-grid arrays are passed to ``prepare_likelihood_inputs``.


Config Quick Reference
----------------------

.. list-table:: Likelihood Parameters
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``Acore``
     - :math:`\sqrt{2}`
     - Core pixel threshold; pixel network energy must exceed
       :math:`2 A_{core}^2 n_{IFO}`
   * - ``netRHO``
     - 4.0
     - Coherent network SNR threshold
   * - ``netCC``
     - 0.5
     - Network correlation threshold (applied to :math:`C_r` in the sky scan)
   * - ``delta``
     - 0.5
     - 2-detector sky location regulator :math:`\in [-1, 1]`;
       :math:`\mathrm{REG}[0] = \min(|\delta|,1)\sqrt2`; :math:`\delta < 0`
       uses :math:`L` for the sky probability
   * - ``cfg_gamma``
     - 0.5
     - :math:`\times`-component regulator threshold
       :math:`\gamma_{reg} = \tfrac23\gamma^2`, :math:`\gamma \in [-1, 1]`;
       :math:`\gamma < 0` adds an antenna prior to the sky probability
   * - ``xgb_rho_mode``
     - false
     - Use :math:`\rho_0 = \sqrt{E_c / (1 + \chi^2(\max(1,\chi^2) - 1))}`
       instead of :math:`\rho = \sqrt{E_c R_c / 2}`
   * - ``healpix``
     - 7
     - Sky map HEALPix resolution (:math:`12 \times 4^7` pixels)
   * - ``upTDF``
     - 4
     - Upsample factor for the time-delay filter rate:
       ``TDRate = rateANA * upTDF``
   * - ``TDSize``
     - 12
     - Time-delay filter half-size in analysis-rate samples (max 20); extended
       if needed to cover the maximum network delay
   * - ``Search``
     - ``""``
     - ``CBC`` / ``BBH`` / ``IMBHB``: one of the conditions for the native
       chirp-mass estimate (also needs ``xgb_rho_mode``,
       ``execution_profile.native_chirp``, ``optim: false``, ``cfg_search``)
   * - ``optim``
     - false
     - cWB optimal-resolution switch; in likelihoodWP it only disables the
       native chirp estimator and controls whether the event ``rate`` is
       exported
   * - ``precision``
     - 0 (disabled)
     - Big-cluster coarse-grid scan: ``csize = |precision| % 65536``; needs
       coarse-grid sky arrays
   * - ``sky_mask``
     - *(none)*
     - Restrict sky scan region (see :ref:`targeted_search`)


Likelihood Pipeline Flow
------------------------

.. code-block:: text

   prepare_likelihood_inputs()                ← once per job segment: delays ml,
        │                                       FP/FX, thresholds, sky mask
        ▼
   per lag: coherence → supercluster → fragment_cluster.clusters
        │
        └── for each cluster with cluster_status <= 0:
              evaluate_cluster_likelihood()
                ├── select_likelihood_pixels()           ← keep loudest BATCH pixels
                ├── compute_dpf_regulator()              ← REG[1] from network index
                ├── scan_sky()                           ← Numba scan over delay groups
                │     ├── time-delay lookup (ml)
                │     ├── DPF + regulated projection
                │     └── netCC gate on C_r, AA = L·C_o → l_max
                ├── compute_statistics_at_sky_position() ← Ec, Rc, χ², ρ at l_max
                ├── get_likelihood_rejection_reason()    ← Lm, Eo−Eh, ρ vs netRHO, N_eff cuts
                ├── populate_detection_statistics()      ← per-pixel stats, MRA waveform
                │                                          energies, netcc, rho
                ├── chirp update                         ← chirp_hough / chirp_micropixel
                └── populate_sky_localization()          ← sky probability, error regions
                      │
                      ▼
              workflow output: Event.output_py() → save_trigger() → Catalog.add_triggers()

:py:func:`~pycwb.modules.likelihoodWP.likelihood.evaluate_fragment_clusters`
is a convenience wrapper for interactive use: it calls
``prepare_likelihood_inputs`` itself and then
:py:func:`~pycwb.modules.likelihoodWP.likelihood.evaluate_cluster_likelihood`
for every surviving cluster of every lag.

This is the Python workflow analogue of the cWB-2G cluster loop:

.. code-block:: text

   read surviving supercluster
        │
        ├── attach time-delay amplitudes to pixels
        ├── run coherent likelihood / sky scan
        ├── reconstruct waveform and event statistics
        └── write trigger and postproduction inputs


.. raw:: html

   <span id="validation-checks"></span>

Inspect reconstructed candidates
--------------------------------

Use the catalog's ranking, correlation and sky quantities together with the
reconstructed waveforms to understand which candidates passed the configured
cuts.

To study a regulator or search-mode change, compare the same background and
injection population under both settings. Inspect which candidates pass the
cuts, how their sky estimates change and which signals are recovered. See
:doc:`tutorial_comparisons` for a worked comparison.

----

**See also:** :doc:`clustering_algorithm` · :doc:`pipeline_lifecycle` · :doc:`postproduction_background`

**Next:** :doc:`event_output` — what the search writes for each trigger
