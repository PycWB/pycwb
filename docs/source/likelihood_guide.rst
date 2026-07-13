.. _likelihood_guide:

Likelihood
==========

.. rubric:: Pipeline: :doc:`data <pipeline_lifecycle>` → :doc:`segments <job_control>` → :doc:`conditioning <pipeline_lifecycle>` → :doc:`WDM <pipeline_lifecycle>` → :doc:`pixels <clustering_algorithm>` → :doc:`clusters <clustering_algorithm>` → **[likelihood]** ← you are here → :doc:`events <pipeline_lifecycle>` → :doc:`bkg <postproduction_background>` → :doc:`ranking <postproduction_xgboost>` → :doc:`eff <postproduction_efficiency>`

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

**Antenna pattern projections** (per sky direction, per detector):

.. math::

   f &= \text{rms} \cdot F_+ \\
   F &= \text{rms} \cdot F_\times

where :math:`\text{rms}` is the root-mean-square of the detector data.

**Optimal polarization angle** :math:`\psi`:

.. math::

   ff &= f \cdot f, \quad FF = F \cdot F, \quad fF = F \cdot f \\[4pt]
   \sin 2\psi &= \frac{2 fF}{\sqrt{(ff - FF)^2 + (2 fF)^2}} \\[4pt]
   \cos 2\psi &= \frac{ff - FF}{\sqrt{(ff - FF)^2 + (2 fF)^2}}

**Effective plus-polarization response**:

.. math::

   |f_+|^2 = \frac{ff + FF + \sqrt{(ff - FF)^2 + (2fF)^2}}{2}

The DPF maximizes the signal projected onto the :math:`+` polarization,
simplifying the likelihood to a single-polarization detection problem.

**DPF Regulator** (:math:`\gamma_{reg}`):

.. math::

   \gamma_{reg} = \gamma^2 \cdot \frac{2}{3}

This penalizes sky directions where :math:`|f_+|` is small (i.e., where the
detector network has poor sensitivity). The regulated energy is:

.. math::

   E_{reg} = \left(\frac{N_{sky}^2}{n_{valid}^2} - 1\right) \cdot E_{threshold}

where :math:`n_{valid}` counts sky directions with :math:`|f_+| > \gamma_{reg}`.


Sky Scan
--------

The sky scan (:py:func:`pycwb.modules.likelihoodWP.sky_scan.scan_sky_for_best_fit`)
is the computational core of the likelihood pipeline. For each HEALPix sky
direction (parallelized with Numba ``prange``):

1. **Apply time delay**: Shift detector data according to the sky-dependent
   time-of-arrival differences:

   .. math::

      \text{data}(t) = TD[t + ml[i, sky]]

   where ``ml[i, sky]`` is the precomputed time-delay lookup table.

2. **Project onto DPF**: Decompose the time-delayed data into signal and
   null components:

   .. math::

      GW = \frac{f_+ \cdot data}{|f_+|^2}

3. **Orthogonalize** signal amplitudes across detectors.

4. **Compute coherent statistics** (see below).

The sky direction that maximizes the sky statistic :math:`L` (or :math:`L_r`
with regularization) is selected as the best-fit location.


SNR Definitions
---------------

pycWB supports two SNR definitions, selectable via ``xgb_rho_mode``:

**cWB SNR** (:math:`\rho`):

.. math::

   \rho = \sqrt{E_c - N_n}

where:

- :math:`E_c` = core coherent energy (sum of signal-component energies)
- :math:`N_n` = null energy (energy orthogonal to the signal model)

This is the standard cWB statistic, representing the excess coherent power
after subtracting the estimated noise.

**XGBoost** :math:`\rho_0`:

.. math::

   \rho_0 = \sqrt{E_c}

Used when ``xgb_rho_mode: true``. This is the coherent energy without null
subtraction, preferred for machine-learning ranking because it preserves more
information for the classifier.

**Network Correlation** (:math:`cc`):

Normalized cross-correlation between detectors, measuring how well the data
matches the expected signal model across the network. Threshold: ``netCC``
(default 0.5).

**Chi-squared** (:math:`\chi^2`):

.. math::

   \chi^2 = \frac{N_{null} + G_n}{N_{pix}^{eff} \cdot n_{IFO}}

where :math:`N_{null}` is the null energy, :math:`G_n` is the noise energy,
:math:`N_{pix}^{eff}` is the effective number of pixels, and :math:`n_{IFO}`
is the number of interferometers. :math:`\chi^2 \sim 1` for well-modeled
signals and :math:`\chi^2 > 1` for glitches or poorly modeled events.


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

   A_{reg} = |\delta| \cdot \sqrt{2} \quad \text{if } |\delta| < 1, \text{ else } 1

``cfg_gamma`` Regulator (Sky Location):

Regulates sky directions where :math:`|f_\times| \ll |f_+|` (i.e., where one
polarization is strongly suppressed). Range :math:`[-1, 1]`, default 0.5.


Detection Statistics
--------------------

After the sky scan, detection statistics are populated per pixel and per
cluster via
:py:func:`pycwb.modules.likelihoodWP.detection_statistics.populate_detection_statistics`:

- **Core flags**: which pixels exceed the :math:`A_{core}` threshold
- **Energy arrays**: signal, null, and noise energies per pixel
- **Subnetwork statistic**:

  .. math::

     E_{sub} = \sum_{i \neq \max} SNR_i \cdot \left(1 + 2 \cdot cc \cdot \frac{E_{sub}}{E_{max}}\right)

- **Multi-resolution analysis (MRA) waveform**: reconstructed strain
  :math:`h(t)` and :math:`h_{rss}` in physical units

The MRA/XTalk catalog used here is not the WDM transform itself. WDM defines
the time-frequency basis; the catalog stores sparse overlaps between pixels at
different WDM resolutions and quadratures. The likelihood/reconstruction path
uses those overlaps to remove duplicated support between resolutions before
synthesizing the final detector waveforms.

**Chirp Mass**: When ``Search`` is set to ``CBC``, ``BBH``, or ``IMBHB``,
the algorithm computes the chirp mass

.. math::

   \mathcal{M} = \frac{(m_1 m_2)^{3/5}}{(m_1 + m_2)^{1/5}}

from the time-frequency evolution of the reconstructed signal.


Big Cluster Optimization
------------------------

When a supercluster contains a large number of pixels, the full-resolution sky
scan becomes expensive. The ``precision`` parameter enables a two-stage
approach:

.. math::

   \text{csize} = \text{precision} \bmod 65536

If :math:`n_{pixels} > nRES \times \text{csize}`, the algorithm uses a coarser
HEALPix sky grid for an initial pass, then refines around promising directions.


Usage Guidance
--------------

For standard users
~~~~~~~~~~~~~~~~~~

The three parameters you should set:

- ``netRHO``: coherent SNR threshold. Lower = more triggers, more background.
  Typical range: 3.5–5.0 for bursts.
- ``netCC``: network correlation threshold. Lower = more triggers, more
  glitches. Typical: 0.4–0.6.
- ``healpix``: sky resolution. Higher = better localization but slower.
  Typical: 6–8.

Everything else should be left at defaults unless you have a specific reason
to change them.

For advanced users
~~~~~~~~~~~~~~~~~~

Tune these only with caution—they can degrade performance if set incorrectly:

- ``delta``: 2-detector sky regulator. Reduces spurious triggers at degenerate
  sky locations. Values near ±1 give stronger regularization. Default 0.5 is
  a good balance.
- ``cfg_gamma``: suppresses sky directions where one polarization is weak.
  Only relevant for networks with similar antenna patterns.
- ``precision``: big-cluster optimization. Controls when to switch to a
  coarser sky grid for very large clusters (``csize = precision % 65536``).
- ``xgb_rho_mode``: use :math:`\rho_0` instead of :math:`\rho`. Set to
  ``true`` when using XGBoost ranking (preserves more information for the
  classifier).
- ``Search``: set to ``CBC`` / ``BBH`` / ``IMBHB`` to enable chirp mass
  computation. Adds computational cost.

Developer notes
~~~~~~~~~~~~~~~

.. admonition:: Implementation detail
   :class: note

   The likelihood pipeline is the most computationally intensive part of pycWB.
   Key implementation details:

   - Sky scan uses Numba ``@njit(parallel=True)`` with ``prange`` over sky
     directions.
   - DPF projection and coherent statistics are computed per sky direction via
     precomputed time-delay arrays (``ml``, ``FP``, ``FX``).
   - JAX device buffers must be explicitly freed after each lag to prevent
     memory leaks—this is a known pitfall.
   - Big-cluster optimization (``bBB``) uses a two-stage approach: coarse sky
     grid first, then refine around promising directions.


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
     - Core pixel amplitude threshold
   * - ``netRHO``
     - 4.0
     - Coherent network SNR threshold
   * - ``netCC``
     - 0.5
     - Network correlation threshold
   * - ``delta``
     - 0.5
     - 2-detector sky location regulator :math:`\in [-1, 1]`
   * - ``cfg_gamma``
     - 0.5
     - :math:`|f_\times| \ll |f_+|` sky location regulator :math:`\in [-1, 1]`
   * - ``xgb_rho_mode``
     - false
     - Use :math:`\rho_0 = \sqrt{E_c}` instead of :math:`\rho = \sqrt{E_c - N_n}`
   * - ``healpix``
     - 7
     - Sky map HEALPix resolution (:math:`12 \times 4^7` pixels)
   * - ``upTDF``
     - 4
     - Upsample factor for time-delay filter rate
   * - ``TDSize``
     - 12
     - Time-delay filter size (max 20)
   * - ``Search``
     - ``""``
     - ``CBC`` / ``BBH`` / ``IMBHB`` to enable chirp mass computation
   * - ``optim``
     - false
     - Optimal resolution likelihood analysis
   * - ``precision``
     - *(computed)*
     - Big-cluster management: ``csize = precision % 65536``
   * - ``sky_mask``
     - *(none)*
     - Restrict sky scan region (see :ref:`targeted_search`)


Likelihood Pipeline Flow
------------------------

.. code-block:: text

   fragment_clusters
        │
        ▼
   prepare_likelihood_inputs()   ← segment-level precomputation (once)
        │
        ▼
   evaluate_fragment_clusters()  ← per-cluster loop
        │
        ├── scan_sky_for_best_fit()        ← Numba-parallel sky scan
        │     └── compute_statistics_at_sky_position()
        │           ├── DPF projection
        │           ├── time-delay apply
        │           └── coherent statistics
        │
        └── populate_detection_statistics() ← per-pixel stats, waveform, chirp mass
              │
              ▼
         Trigger output (Parquet catalog + JSON)

This is the Python workflow analogue of the cWB-2G cluster loop:

.. code-block:: text

   read surviving supercluster
        │
        ├── attach time-delay amplitudes to pixels
        ├── run coherent likelihood / sky scan
        ├── reconstruct waveform and event statistics
        └── write trigger and postproduction inputs


Validation Checks
-----------------

After configuring likelihood parameters, verify:

- **SNR distribution is reasonable**: peak near ``netRHO``, smooth tail to
  high SNR. A sharp cutoff at ``netRHO`` without a tail suggests the
  threshold is too aggressive.
- **Sky positions are physically distributed**: for all-sky searches, triggers
  should cover the full sky (modulo antenna pattern sensitivity). Clustering
  at one sky location suggests a detector artifact or xtalk.
- **:math:`\chi^2` distribution is well-behaved**: :math:`\chi^2 \sim 1` for
  the bulk of triggers. A long tail of high :math:`\chi^2` indicates glitch
  contamination.
- **``delta`` and ``cfg_gamma`` aren't causing sky bias**: for 2-detector
  networks, check that the sky distribution of triggers isn't artificially
  peaked at the degenerate sky locations that the regulators are meant to
  suppress.


----

**See also:** :doc:`clustering_algorithm` · :doc:`pipeline_lifecycle` · :doc:`postproduction_background`

**Next:** :doc:`postproduction_background` — how background is estimated from lags
