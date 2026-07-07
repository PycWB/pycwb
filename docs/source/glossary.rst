.. _glossary:

Glossary
========

.. glossary::
   :sorted:

   background
      The accidental coincidence rate—how often noise fluctuations produce
      triggers at a given ranking statistic. Estimated from **non-zero lag**
      data where real signals are broken by time-shifting one detector.

   burst
      A short-duration gravitational-wave signal without a precise waveform
      model. Sources include supernovae, gamma-ray bursts, and neutron star
      mergers.

   cWB
      **Coherent Wave Burst** — the core algorithm that pycWB implements. Uses
      wavelet transforms and network coherence to detect unmodeled bursts.

   cc
      **Network correlation** — normalized cross-correlation between detectors
      measuring how well data matches the expected signal model.
      Threshold: ``netCC`` (default 0.5).

   chi2
      (:math:`\chi^2`) — goodness-of-fit statistic. :math:`\chi^2 \sim 1` for
      well-modeled signals, :math:`\chi^2 > 1` for glitches.

   cluster
      A group of time-frequency pixels with excess coherent power, selected
      before likelihood evaluation. Multiple clusters are merged into
      **superclusters**.

   coherent
      Appearing consistently across all detectors after accounting for
      time-of-flight delays. Real GW signals are coherent; instrumental noise
      generally is not.

   config
      Short for ``user_parameters.yaml`` — the single YAML file controlling
      all aspects of a pycWB search.

   DPF
      **Dominant Polarization Frame** — the optimal polarization basis for a
      given sky direction. Maximizes the signal projected onto the
      :math:`+` polarization, simplifying likelihood to single-polarization
      detection.

   DQ
      **Data Quality** — flags indicating whether detector data is suitable
      for science analysis. CAT0 = science mode, CAT1 = bad data (vetoed),
      CAT2 = marginal data (applied as veto windows).

   event
      A candidate gravitational-wave trigger produced by the likelihood
      pipeline. Includes time, frequency, sky position, SNR, and ranking
      statistic.

   FAR
      **False Alarm Rate** — expected rate of background triggers at or above
      a given ranking statistic. :math:`FAR(\rho^*) = N_{bkg}(\rho \ge \rho^*) / T_{bkg}`.
      Units: yr\ :sup:`-1`.

   HEALPix
      Hierarchical Equal Area isoLatitude Pixelation — a scheme for
      pixelizing the sphere. ``healpix`` parameter controls sky resolution:
      :math:`N_{pix} = 12 \times 4^{healpix}`.

   hrss
      (:math:`h_{rss}`) — root-sum-squared strain amplitude of a signal.
      Used as the amplitude measure for burst injections and efficiency.

   hrss50 / hrss90
      The :math:`h_{rss}` at which 50% / 90% of injected signals are
      recovered. Standard figures of merit for search sensitivity.

   injection
      A simulated gravitational-wave signal added to detector data to measure
      search sensitivity. Defined by waveform parameters and sky location.

   job
      One unit of computation: a specific segment processed at a specific
      lag for a specific injection trial. Submitted to a cluster as an
      independent task.

   job segment
      A GPS time window containing detector data. A search is split into many
      segments that can run in parallel. Controlled by ``segLen``,
      ``segMLS``, ``segEdge``.

   lag
      A time-shift applied to one detector's data relative to others. Used
      to estimate the **background**: non-zero lags break real signal
      coincidences, producing only accidental triggers. Controlled by
      ``lagSize``, ``lagStep``, ``lagOff``, ``lagMax``.

   lagOff
      Offset for the first lag index. ``lagOff = 0`` includes the zero-lag
      (physical coincidence). ``lagOff > 0`` skips the first N lags.

   likelihood
      The mathematical framework that evaluates how likely a cluster of pixels
      is to be a real GW signal rather than noise. Produces SNR, network
      correlation, and sky localization.

   livetime
      Total analyzed time used for background estimation.
      :math:`T_{bkg} = N_{jobs} \times (N_{lags} - 1) \times T_{seg}`.

   MRA
      **Multi-Resolution Analysis** — using wavelet transforms at multiple
      frequency resolutions to capture signals of different durations.

   network
      The set of gravitational-wave detectors used in a search. Common
      networks: LH (LIGO Hanford + Livingston), LHV (LIGO + Virgo).

   pixel
      A single time-frequency cell in the wavelet transform output. Excess
      power pixels are selected above a threshold and grouped into clusters.

   postproduction
      The pipeline that runs **after** all search jobs finish. Produces
      background estimates, XGBoost ranking, detection efficiency curves,
      and HTML reports.

   progress
      Per-job metadata (Parquet format) tracking which jobs/lags ran, how
      long each took, and whether they succeeded. Used by postproduction.

   ranking statistic
      The final score assigned to each event, combining multiple features
      (SNR, :math:`\chi^2`, correlation, etc.) via XGBoost. Used for FAR
      assignment and sensitivity.

   rho
      (:math:`\rho`) — coherent network SNR. :math:`\rho = \sqrt{E_c - N_n}`
      where :math:`E_c` is coherent energy and :math:`N_n` is null energy.

   rho0
      (:math:`\rho_0`) — unsubtracted SNR: :math:`\rho_0 = \sqrt{E_c}`.
      Used when ``xgb_rho_mode: true``.

   segment
      See **job segment**.

   simulation
      An injection study measuring how efficiently the search recovers
      signals. Requires matched injection truth tables.

   sky mask
      A restricted sky region for the likelihood scan. Used in targeted
      searches (e.g., GRB follow-up). Types: UniformAllSky, Patch, Fixed,
      Custom.

   slag
      **Super lag** — a segment-level time shift for multi-detector networks.
      Increases total jobs by factor of ``slagSize``. Controlled by
      ``slagSize``, ``slagMin``, ``slagMax``, ``slagOff``.

   SNR
      **Signal-to-Noise Ratio** — measure of signal strength relative to
      background noise. See **rho**.

   subnetwork cut
      Per-sky-direction threshold cuts that remove pixel clusters unlikely
      to be astrophysical. Controlled by ``subnet``, ``subcut``, ``subrho``.

   supercluster
      A merged group of nearby pixel clusters. The superclustering step
      determines which pixel groups are treated as a single candidate.
      Controlled by ``TFgap``, ``Tgap``, ``Fgap``.

   time-frequency map
      The output of the wavelet transform: a 2D representation of signal
      power vs. time and frequency. Excess pixels are selected from this map.

   trial / trial_idx
      A group of injections sharing the same noise background within a job
      segment. ``trial_idx`` identifies which trial an injection belongs to.

   WDM
      **Wilson-Daubechies-Meyer** wavelet — the specific wavelet family used
      by cWB for time-frequency decomposition.

   whitening
      Normalizing the detector noise to have flat (white) frequency spectrum.
      Methods: ``wavelet``, ``mesa``, or ``mixed``.

   XGBoost
      eXtreme Gradient Boosting — the machine-learning classifier used to
      combine multiple event features into a single ranking statistic.

   XTalk
      **Cross-talk** — coherent instrumental artifacts that mimic GW signals.
      pycWB includes XTalk identification and subtraction.

   zero-lag
      The unshifted (physical) coincidence between detectors where a real
      GW signal would appear. Excluded from background estimation.
