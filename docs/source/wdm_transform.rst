.. _wdm_transform:

WDM Time-Frequency Transform
============================

.. stage-nav:: search
   :current: wdm

This guide explains how pycWB converts each conditioned detector strain into
Wilson-Daubechies-Meyer (WDM) time-frequency maps at several resolutions. It
also covers how the time-delay filters and the cross-talk (XTalk) catalog are
prepared, and which parameters control them.

.. contents:: Table of Contents
   :depth: 2
   :local:


Why this matters
----------------

All later stages work on WDM pixels: pixel selection, clustering, the
likelihood sky scan and waveform reconstruction. ``l_low`` … ``l_high`` sets
the available pixel shapes, from short and broadband to long and narrowband,
and the WDM filter parameters must match the XTalk catalog. For a standard
search keep the defaults; change the levels only with a catalog that covers them.


The WDM Basis
-------------

The WDM transform (Necula et al., *J. Phys.: Conf. Ser.* 363 (2012) 012032)
tiles the time-frequency plane evenly. With :math:`M` layers, :math:`N`
samples at rate :math:`R` become a complex array of :math:`M+1` frequency
layers (:math:`m = 0` at DC to :math:`m = M` at :math:`R/2`, layer :math:`m`
centred at :math:`m\,\Delta f`) by :math:`\lceil N/M \rceil` time bins. The
real part holds the 0° (``00``) amplitude, the imaginary part the 90° (``90``)
quadrature. Every pixel has the same area:

.. math::

   \Delta t = \frac{M}{R}, \qquad \Delta f = \frac{R}{2M}, \qquad
   \Delta t\,\Delta f = \frac{1}{2}.

Resolution levels
~~~~~~~~~~~~~~~~~

pycWB sets :math:`R` to the analysis rate ``rateANA`` (``fResample`` if set,
otherwise ``inRate``, divided by :math:`2^{\text{levelR}}`; see
:doc:`data_conditioning`). Level :math:`l` uses :math:`M_l = 2^l` layers:

.. math::

   \Delta t_l = \frac{2^l}{\text{rateANA}}, \qquad
   \Delta f_l = \frac{\text{rateANA}}{2^{l+1}}, \qquad
   r_l = \frac{1}{\Delta t_l} = \frac{\text{rateANA}}{2^l}.

:math:`r_l` is the pixel ``rate`` used by the clustering metric. The defaults
(``inRate`` 16384, ``levelR`` 2, so ``rateANA`` = 4096 Hz; levels 3 … 8) give:

=====  =====  ============  ============  ===========
Level  M      Δf [Hz]       Δt [ms]       rate [Hz]
=====  =====  ============  ============  ===========
3      8      256           1.953         512
4      16     128           3.906         256
5      32     64            7.8125        128
6      64     32            15.625        64
7      128    16            31.25         32
8      256    8             62.5          16
=====  =====  ============  ============  ===========

WDM filter
~~~~~~~~~~

The basis uses a Meyer window with transition parameter :math:`K = M` and
beta-function order ``WDM_beta_order``. The one-sided filter length
:math:`m_H` is the smallest :math:`N` with :math:`N-1` a multiple of
:math:`2M`, :math:`N \ge 6M+1`, and truncated-tail energy below
:math:`10^{-p}`, :math:`p` = ``WDM_precision``. The defaults (6, 10) give
:math:`m_H = 12M+1` taps (0.75 s at level 8, ``rateANA`` = 4096 Hz). The filter
must fit in the padding: :math:`m_H/\text{rateANA} \le` ``segEdge``.


Multi-Resolution Analysis
-------------------------

pycWB transforms the same whitened strain independently at every level from
``l_high`` down to ``l_low`` (``nRES`` = ``l_high`` − ``l_low`` + 1; this
descending order is ``config.WDM_level``). A short broadband signal is compact
at low :math:`l`, a long narrowband one at high :math:`l`. Pixel selection and
single-resolution clustering run per level; superclustering merges across
levels (see :doc:`clustering_algorithm`). Bases at different levels are not
orthogonal, so the same energy appears in several maps; the XTalk catalog
records these overlaps (see `XTalk (MRA) Catalog`_).

**Time-index parity.** WDM basis functions alternate with time-index parity,
and the XTalk catalog is indexed by it, so shifts must keep parity at every
level. :py:meth:`~pycwb.config.config.Config.check_lagStep` requires
``lagStep``, ``segEdge`` and ``segMLS`` to be multiples of :math:`2/r_{\min}`,
:math:`r_{\min} = \text{rateANA}/2^{l_{\rm high}}` (0.125 s by default); it
tests that :math:`\lfloor x\,r_{\min} \rfloor` is even. Job construction
shortens a job by 1 s if its length breaks parity (see :ref:`job_control`).


Time-Delay Filters
------------------

Likelihood needs pixel amplitudes at sky-dependent detector delays. WDM gives
them from the map itself, without re-transforming shifted data (Necula et al.
2012, Sec. 4). ``WDM.set_td_filter(TDSize, upTDF)`` precomputes the tables:

.. math::

   \tau = d\,\delta\tau, \qquad
   \delta\tau = \frac{1}{\text{rateANA}\cdot\text{upTDF}} = \frac{1}{\text{TDRate}},
   \qquad |d| \le J = M\cdot\text{upTDF},

covering :math:`|\tau| \le \Delta t`; longer delays become an even number of
whole time bins plus an in-table remainder. The delayed amplitude of pixel
:math:`(n, m)` combines time bins :math:`n-\text{TDSize} \dots n+\text{TDSize}`
of layers :math:`m-1, m, m+1`; :math:`M` must be even. ``segEdge`` must be at least
:math:`\lfloor 1.5\,\text{TDSize}/r_l + 0.5 \rfloor` s to avoid distorted
border pixels. The delays extracted per pixel, :math:`K_{td}`, are set in
supercluster setup (see :doc:`clustering_algorithm`, "Time-Delay
Precomputation").


XTalk (MRA) Catalog
-------------------

The catalog is ``wdmXTalk`` under ``filter_dir`` (full path: derived key
``MRAcatalog``). For each pair of resolutions, including a resolution with
itself, it is indexed by the frequency layer and time parity of the first
pixel. Each entry is an overlapping pixel of the second resolution: a relative
index plus four overlaps (00·00, 00·90, 90·00, 90·90). pycWB only reads
catalogs; the cWB C++ ``WDMOverlap`` builder (``cwb-core/``) stores overlaps
above a threshold (default 0.01), so the table is sparse. Pixels name their
resolution by :math:`M+1` (the ``layers`` field, cWB convention).

The header stores ``nRes`` and the layer counts; a *tagged* catalog also
stores the beta order, precision and :math:`K` used to build it. At
configuration load, :py:meth:`~pycwb.config.config.Config.check_MRA_catalog`
requires every level's :math:`2^l` in the layer list, and a tagged catalog
overwrites ``WDM_beta_order`` and ``WDM_precision`` (an untagged one is not
checked against them).

The default catalog (layers 8 … 256, per its name) matches levels 3 … 8. A
missing file is downloaded by :py:meth:`~pycwb.config.config.Config.check_xtalk_file`
if its name is in the PycWB ``xtalk-data`` repository; otherwise
``FileNotFoundError``. :py:meth:`pycwb.modules.xtalk.type.XTalk.load` parses a
``.bin``/``.xbin`` file and writes a ``.npz`` with the same stem beside it,
which later loads read instead.

Later, the sub-network cut uses the catalog in its MRA step (see
:doc:`clustering_algorithm`), and likelihood uses the per-cluster overlap lists
from :py:meth:`~pycwb.modules.xtalk.type.XTalk.get_xtalk_pixels` for
cross-talk-corrected data and null energies and to normalise the packet
amplitudes stored on the cluster pixels (see :doc:`likelihood_guide`). Waveform
synthesis reads those amplitudes, not the catalog itself.


Implementation
--------------

- :py:func:`pycwb.modules.coherence_native.setup.setup_coherence` builds
  ``wdm_wavelet.wdm.WDM(M, K=M, beta_order, precision)`` (companion package
  ``wdm-wavelet``) per level and transforms all detectors in one JAX call,
  :py:func:`pycwb.modules.coherence_native.tf_batch_generation.batch_t2w_detectors`
  (per-detector fallback on failure), into
  :py:class:`pycwb.types.time_frequency_map.TimeFrequencyMap` objects.
- :py:func:`pycwb.utils.td_vector_batch.build_td_inputs_cache` builds a second
  WDM for every level, calls ``set_td_filter(TDSize, upTDF)`` and transforms
  each detector's strain again. It stores zero-padded float32 00/90 planes and
  the filter tables as :py:class:`pycwb.types.td_batch_inputs.TDBatchInputs`.
- :py:class:`pycwb.modules.xtalk.type.XTalk` and
  :py:func:`pycwb.modules.xtalk.monster.read_catalog_metadata` read the
  catalog; :py:meth:`pycwb.config.config.Config.add_derived_key` derives
  ``rateANA``, ``nRES``, ``WDM_level``, ``TDRate`` and ``MRAcatalog``.
- ``_create_wdm_set_python`` (:py:mod:`pycwb.modules.reconstruction.getMRAwaveform`)
  builds the filters likelihood uses for waveform synthesis and raises
  ``ValueError`` if a ``segEdge`` limit is violated;
  :py:func:`pycwb.modules.multi_resolution_wdm.wdm.create_wdm_for_level` does
  the same for the ROOT modules with :py:class:`pycwb.types.wdm.WDM`.

Execution-profile flags (``tiled_wdm``, ``wdm_*``) change how the transform
runs, not the analysis (see :ref:`execution_profile_options`).

cWB-2G correspondence
~~~~~~~~~~~~~~~~~~~~~

This is the "WDM and MRA setup" row of the :ref:`pipeline_lifecycle` table:

- ``WDM<double>(layers, layers, WDM_beta_order, WDM_precision)`` per level →
  ``wdm_wavelet.wdm.WDM(M, M, ...)`` (:py:class:`pycwb.types.wdm.WDM` on ROOT).
- ``WDM::setTDFilter(TDSize, upTDF)`` → ``set_td_filter(TDSize, upTDF)``.
- ``network::setMRAcatalog(MRAcatalog)`` (``monster`` catalog) →
  :py:meth:`~pycwb.modules.xtalk.type.XTalk.load` of ``config.MRAcatalog``.


Configuration
-------------

Defaults come from :py:mod:`pycwb.constants.user_parameters_schema`.

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Parameter
     - Default
     - Meaning
   * - ``l_low``
     - 3
     - Lowest level (:math:`M = 2^{l_{\rm low}}`, finest time resolution)
   * - ``l_high``
     - 8
     - Highest level (:math:`M = 2^{l_{\rm high}}`, finest frequency resolution)
   * - ``WDM_beta_order``
     - 6
     - Beta-function order of the Meyer window (overridden by a tagged catalog)
   * - ``WDM_precision``
     - 10
     - Filter truncation precision, :math:`10^{-p}` (overridden by a tagged catalog)
   * - ``TDSize``
     - 12 (max 20)
     - Half-width of the time-delay filters, in time bins
   * - ``upTDF``
     - 4
     - Delay upsampling factor; ``TDRate`` = ``rateANA`` × ``upTDF``
   * - ``wdmXTalk``
     - ``wdmXTalk/OverlapCatalog_Lev_8_16_32_64_128_256_iNu_4_Prec_10.bin``
     - XTalk catalog path relative to ``filter_dir``
   * - ``filter_dir``
     - ``""``
     - Catalog directory; if empty, ``$HOME_WAT_FILTERS``, else the current directory
   * - ``segEdge``
     - 8.0
     - Segment padding [s]; must contain the WDM and TD filter lengths

``rateANA``, ``nRES``, ``WDM_level``, ``TDRate`` and ``MRAcatalog`` are
derived; do not set them (see :ref:`schema`).


Output
------

- **TF maps**: one complex :math:`(M+1) \times n_{\rm time}`
  ``TimeFrequencyMap`` per detector and level (0 to :math:`R/2`; band limits
  and ``segEdge`` are metadata). ``setup_coherence`` passes them straight to
  pixel selection (see :doc:`clustering_algorithm`, "Pixel Selection") and
  returns per-level dicts (``level``, ``layers`` = :math:`M`, ``rate`` =
  :math:`r_l`, ``tf_maps``, ``Eo``) for ``coherence_single_lag``.
- **TD-inputs cache**: ``{M: [TDBatchInputs per detector]}``, also keyed by
  :math:`M+1`. Supercluster uses it to fill delayed amplitudes.
- **XTalk object**: passed to the sub-network cut and to likelihood.


Validation Checks
-----------------

- **Resolution log**: each level logs ``rate(hz)``, ``layers``, ``df(hz)`` and
  ``dt(ms)``; compare with the table for your ``rateANA``.
- **Catalog match**: ``analysis layers do not match the MRA catalog``: a level
  is missing from the catalog (see :ref:`troubleshooting`).
- **Filter parameters**: if you change ``WDM_beta_order`` or
  ``WDM_precision``, use a catalog built with the same values. A tagged
  catalog takes precedence over YAML; the log then shows ``updating beta
  order and precision from MRA catalog``.
- **Edge length**: ``Filter length must be <= segEdge`` or ``segEdge must be >
  1.5x the length for time delay amplitudes`` means ``segEdge`` is too short
  for ``l_high`` or ``TDSize``.
- **Parity**: ``not a multple of 2*max_time_resolution`` (spelled so in the
  code) names the offending ``lagStep``, ``segEdge`` or ``segMLS``.
- **Catalog cache**: if you replace a ``.bin`` catalog, delete the ``.npz``
  beside it; the ``.npz`` is loaded in preference.


----

**See also:** :doc:`pipeline_lifecycle` · :doc:`data_conditioning` · :doc:`clustering_algorithm`

**Next:** :doc:`clustering_algorithm` — pixel selection, clustering and superclustering
