.. _clustering_algorithm:

Clustering Algorithm
====================

.. rubric:: Pipeline: :doc:`data <pipeline_lifecycle>` → :doc:`segments <job_control>` → :doc:`conditioning <pipeline_lifecycle>` → :doc:`WDM <pipeline_lifecycle>` → :doc:`pixels <pipeline_lifecycle>` → **[clusters & superclusters]** ← you are here → :doc:`likelihood <likelihood_guide>` → :doc:`events <pipeline_lifecycle>` → :doc:`bkg <postproduction_background>` → :doc:`ranking <postproduction_xgboost>` → :doc:`eff <postproduction_efficiency>`

This guide describes pycWB's pixel clustering and superclustering algorithms,
including the configurable parameters that control how time-frequency pixels
are grouped into candidate events.

.. contents:: Table of Contents
   :depth: 2
   :local:


Why this matters
----------------

The clustering stage decides which time-frequency pixels are grouped into a
candidate event. Most users do not need to modify clustering parameters for
a standard search, but these settings affect background rate, sensitivity,
and glitch rejection. Tune only if you understand the trade-offs.


Overview
--------

After the coherent wavelet transform identifies excess power pixels in the
time-frequency plane, pycWB groups these pixels into **clusters** and then
merges nearby clusters into **superclusters**. The superclustering step is
critical: it determines which pixel groups are treated as a single
gravitational-wave candidate for likelihood evaluation.

.. mermaid::

   flowchart TD
     A[Selected Pixels] --> B[Per-Resolution Clusters]
     B --> C[Multi-Resolution Merge]
     C --> D[Sub-Network Cut]
     D --> E[Superclusters]
     E --> F[Defragmentation]
     F --> G[To Likelihood]
     B -.->|TFgap| C
     D -.->|subnet / subcut| E
     E -.->|Tgap / Fgap| F

The production clustering code lives in
:py:mod:`pycwb.modules.super_cluster_native`. An experimental
:py:mod:`pycwb.modules.clustering` module exists for future algorithm
development (DBSCAN, HDBSCAN, OPTICS, etc.) but is not yet production-ready.


Pipeline: From Pixels to Fragment Clusters
------------------------------------------

The clustering pipeline proceeds through these steps:

1. **Pixel Selection** — excess power pixels are selected above a threshold
   in the time-frequency plane.
2. **Per-Resolution Clustering** — pixels at each WDM resolution level are
   clustered independently.
3. **Multi-Resolution Merging** — clusters from different resolution levels
   are merged.
4. **Sub-Network Cut** — per-sky-direction threshold cuts are applied to
   remove accidental coincidences.
5. **Superclustering** — nearby clusters in time-frequency are merged into
   superclusters.
6. **Defragmentation** — a final cleanup pass merges any remaining close
   clusters.

The high-level wrapper is
:py:func:`pycwb.modules.super_cluster_native.super_cluster.supercluster_wrapper`,
which calls :py:func:`~.setup_supercluster` once at initialization and
:py:func:`~.supercluster_single_lag` for each lag.


Pixel Selection
---------------

Pixels are selected based on their coherent energy and network correlation.
Key parameters controlling pixel selection:

.. list-table::
   :header-rows: 1
   :widths: 22 15 63

   * - Parameter
     - Default
     - Description
   * - ``bpp``
     - 0.001
     - Black pixel selection probability (fraction of pixels kept)
   * - ``BATCH``
     - 10000
     - Maximum pixels per loadTDamp batch
   * - ``LOUD``
     - 200
     - Pixels per cluster for time-delay amplitude loading
   * - ``pattern``
     - 0
     - Pixel selection pattern: 0 = single pixel, 1–8 = multi-pixel packets, <0 = mixed, >0 = packed


Superclustering Algorithm
-------------------------

The superclustering algorithm
(:py:func:`pycwb.modules.super_cluster_native.super_cluster.supercluster`)
merges pixel clusters that are close in time and frequency:

1. **Build pixel matrix**: For all input clusters, construct a compact feature
   matrix with these columns per pixel:

   - Central time (normalized by rate × layer)
   - Central frequency
   - Inverse rate (:math:`1 / \text{rate}`)
   - Half-rate (:math:`\text{rate} / 2`)
   - Parent cluster ID
   - Per-interferometer pixel indices

2. **Find cluster links**: Using
   :py:func:`~pycwb.modules.super_cluster_native.utils.get_cluster_links`,
   identify pairs of clusters whose pixels are within a time-frequency gap
   threshold.

3. **Union-Find merging**: Linked clusters are merged using a Numba
   JIT-compiled union-find data structure with path compression and
   union-by-rank
   (:py:func:`~pycwb.modules.super_cluster_native.utils.aggregate_clusters_from_links`).

4. **Compute supercluster statistics**: For each merged supercluster,
   calculate combined time, frequency, rate range, energy, and likelihood.

The gap threshold for linking is controlled by ``TFgap``:

.. math::

   \text{linked if } \Delta t < \text{TFgap} \text{ AND } \Delta f \cdot \text{rate} < \text{TFgap}

where :math:`\Delta t` and :math:`\Delta f` are the time and frequency
separations between pixels, and ``rate`` is the WDM analysis rate at that
resolution level.


Sub-Network Cut
---------------

The sub-network cut
(:py:func:`pycwb.modules.super_cluster_native.sub_net_cut.sub_net_cut`) is a
per-sky-direction selection that removes pixel clusters unlikely to be
astrophysical:

- For each sky direction, the algorithm evaluates whether the pixel subnetwork
  exceeds coherence thresholds.
- Two sky arrays are precomputed for efficiency: full resolution (for
  likelihood) and reduced resolution (capped at ``MIN_SKYRES_HEALPIX`` for the
  sub-network cut).
- The cut is Numba-accelerated and handles cross-talk (XTalk) pixel lookups
  internally.

Parameters controlling the sub-network cut:

.. list-table::
   :header-rows: 1
   :widths: 22 15 63

   * - Parameter
     - Default
     - Description
   * - ``subnet``
     - 0.7
     - Sub-network coherence threshold :math:`\in [0, 0.7]`
   * - ``subcut``
     - 0.33
     - Sub-network threshold in sky loop :math:`\in [0, 1]`
   * - ``subnorm``
     - 0.0
     - Sub-network norm threshold (enabled if > 0) :math:`\in [0, 2 \times nRes]`
   * - ``subrho``
     - 0.0
     - Sub-network sky loop rho threshold (≤ 0 → uses ``netRHO``)
   * - ``subacor``
     - 0.0
     - Sub-network sky loop Acore threshold (≤ 0 → uses ``Acore``)
   * - ``select_subnet``
     - 0.1
     - Subnet netcluster selection threshold
   * - ``select_subrho``
     - 5.0
     - Subrho netcluster selection threshold

When ``subrho`` ≤ 0, the standard ``netRHO`` threshold is used for the
sub-network cut. Similarly, ``subacor`` ≤ 0 falls back to ``Acore``.


Defragmentation
---------------

After superclustering, a defragmentation pass
(:py:func:`pycwb.modules.super_cluster_native.super_cluster.defragment`)
merges any remaining clusters that are close in time and frequency:

.. list-table::
   :header-rows: 1
   :widths: 22 15 63

   * - Parameter
     - Default
     - Description
   * - ``Tgap``
     - 3.0 s
     - Defragmentation time gap—clusters within this time are merged
   * - ``Fgap``
     - 130 Hz
     - Defragmentation frequency gap—clusters within this frequency are merged
   * - ``TFgap``
     - 6.0
     - Time-frequency pixel separation threshold for linking


Time-Delay Precomputation
-------------------------

At setup time
(:py:func:`pycwb.modules.super_cluster_native.super_cluster.setup_supercluster`),
the time-delay range is precomputed:

.. math::

   K_{td} = \max(TDSize \times upTDF,\ \lfloor \text{max\_delay} \times TDRate \rfloor + 1)

This determines the number of time-delay samples needed for the sky-dependent
time shifting of detector data during likelihood evaluation.

Related parameters: ``TDSize`` (default 12, max 20), ``upTDF`` (default 4,
upsample factor for TD filter rate).


Usage Guidance
--------------

For standard users
~~~~~~~~~~~~~~~~~~

You likely don't need to change any clustering parameters. The defaults are
chosen for general-purpose burst searches. If you must tune:

- ``TFgap``: increase to merge more pixels (fewer, larger clusters); decrease
  to split clusters (more, smaller events).
- ``Tgap`` / ``Fgap``: control defragmentation. Larger values merge more.

For advanced users
~~~~~~~~~~~~~~~~~~

Tune these only if you understand the impact on background and sensitivity:

- ``subnet``, ``subcut``, ``subrho``, ``subacor``: sub-network cut thresholds.
  Higher values are more selective (fewer clusters, lower background, but may
  reject real signals).
- ``bpp``: black pixel probability. Lower = fewer pixels selected. Affects
  sensitivity to short-duration signals.
- ``pattern``: multi-pixel packet mode. Non-zero values group neighboring
  pixels, which helps for extended signals but can merge distinct events.

Developer notes
~~~~~~~~~~~~~~~

.. admonition:: Implementation detail
   :class: note

   The clustering code is Numba JIT-compiled for performance. Key internals:

   - ``_build_link_pixel_matrix`` constructs the feature matrix from
     ``PixelArrays`` (struct-of-arrays layout).
   - Union-find uses path compression + union-by-rank for :math:`O(\alpha(N))`
     merging.
   - Sub-network cut precomputes two HEALPix sky arrays: full resolution (for
     likelihood) and capped (``MIN_SKYRES_HEALPIX``, for the cut).


Config Quick Reference
----------------------

.. list-table:: Clustering & Superclustering Parameters
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``bpp``
     - 0.001
     - Black pixel selection probability
   * - ``BATCH``
     - 10000
     - Max pixels per loadTDamp batch
   * - ``LOUD``
     - 200
     - Pixels per cluster for TD amplitude loading
   * - ``pattern``
     - 0
     - Pixel selection pattern (0 = single, 1–8 = packets)
   * - ``TFgap``
     - 6.0
     - TF pixel separation for cluster linking
   * - ``Tgap``
     - 3.0
     - Defragmentation time gap [s]
   * - ``Fgap``
     - 130
     - Defragmentation frequency gap [Hz]
   * - ``subnet``
     - 0.7
     - Sub-network threshold
   * - ``subcut``
     - 0.33
     - Sub-network skyloop threshold
   * - ``subnorm``
     - 0.0
     - Sub-network norm threshold
   * - ``subrho``
     - 0.0
     - Sub-network skyloop rho
   * - ``subacor``
     - 0.0
     - Sub-network skyloop Acore
   * - ``select_subnet``
     - 0.1
     - Subnet netcluster selection
   * - ``select_subrho``
     - 5.0
     - Subrho netcluster selection
   * - ``TDSize``
     - 12
     - Time-delay filter size (max 20)
   * - ``upTDF``
     - 4
     - Upsample factor for TD filter rate


Validation Checks
-----------------

After tuning clustering parameters, verify:

- **Number of clusters scales with segment length**: longer segments should
  produce proportionally more clusters. A flat or zero count suggests the
  pixel selection threshold is too strict.
- **Superclusters merge within TFgap**: check that merged superclusters'
  constituent pixels are within ``TFgap`` in time-frequency. Pixels outside
  this range should not be in the same supercluster.
- **Defragmentation doesn't merge independent events**: verify that
  ``Tgap`` and ``Fgap`` are small enough that distinct astrophysical
  signals (e.g., from different sources) are not merged into one event.


----

**See also:** :doc:`pipeline_lifecycle` · :doc:`likelihood_guide` · :doc:`job_control`

**Next:** :doc:`likelihood_guide` — how candidate events are scored
