.. _tutorial_injection:

Step by Step Injection Search
=============================

Intermediate · Prerequisites: :ref:`start_here` and :doc:`tutorial_search`

This walkthrough explains the Python stages and numerical conventions behind
an injection search. For a worked recovery experiment, use
:doc:`tutorial_population`; for configuring your own campaign, use
:doc:`injection_infrastructure`. The stage example below uses ``examples/injection``.

First, create a project folder and copy the example files:

.. code-block:: bash

   mkdir my_search
   cp -r [path_to_source_code]/examples/injection/* my_search/
   cd my_search

Run the full injection search with:

.. code-block:: bash

   pycwb run user_parameters_injection.yaml

The same workflow can be called from Python:

.. code-block:: python

   from pycwb.workflow.run import search

   search("user_parameters_injection.yaml")

To inspect the stages manually, load the configuration and create the job
segments:

.. code-block:: python

   from pycwb.config import Config
   from pycwb.modules.job_segment import create_job_segment_from_config
   from pycwb.modules.logger import logger_init

   logger_init()
   config = Config()
   config.load_from_yaml("user_parameters_injection.yaml")
   job_segments = create_job_segment_from_config(config)
   job_segment = job_segments[0]

For an injection example with generated noise, build the base noise and inject
the configured waveform into each detector stream:

.. code-block:: python

   from pycwb.modules.injection import generate_strain_from_injection
   from pycwb.modules.read_data.simulations import generate_noise_for_job_seg

   data = generate_noise_for_job_seg(job_segment, config.inRate, f_low=config.fLow)

   for injection in job_segment.injections:
       injected = generate_strain_from_injection(
           injection,
           config,
           job_segment.sample_rate,
           job_segment.ifos,
       )
       for i, strain in enumerate(injected):
           data[i].inject(strain, copy=False)

Resample to the analysis rate before conditioning; the example uses
``levelR: 3``. This walkthrough injects before resampling, whereas the segment
workflow resamples target-SNR injections separately (see below). The
data-conditioning module then whitens the data and returns the conditioned
strains and the nRMS maps used by the later stages:

.. code-block:: python

   from pycwb.modules.data_conditioning import condition_strains
   from pycwb.modules.read_data.data_check import check_and_resample_py

   data = [check_and_resample_py(strain, config, i) for i, strain in enumerate(data)]
   strains, nRMS = condition_strains(config, data)

The native production path then performs setup once and reuses it for each
time-slide lag:

.. code-block:: python

   from pycwb.modules.coherence_native.coherence import setup_coherence, coherence_single_lag
   from pycwb.modules.likelihoodWP import prepare_likelihood_inputs, evaluate_cluster_likelihood
   from pycwb.modules.super_cluster_native.super_cluster import setup_supercluster, supercluster_single_lag
   from pycwb.modules.xtalk.type import XTalk
   from pycwb.utils.td_vector_batch import build_td_inputs_cache

   coherence_setup = setup_coherence(config, strains, job_seg=job_segment, nRMS=nRMS)
   td_inputs_cache = build_td_inputs_cache(config, strains)
   supercluster_setup = setup_supercluster(config, gps_time=float(strains[0].start_time))
   likelihood_setup = prepare_likelihood_inputs(
       config,
       strains,
       config.nIFO,
       ml=supercluster_setup.get("ml_likelihood", supercluster_setup["ml"]),
       FP=supercluster_setup.get("FP_likelihood", supercluster_setup["FP"]),
       FX=supercluster_setup.get("FX_likelihood", supercluster_setup["FX"]),
   )
   xtalk = XTalk.load(config.MRAcatalog)

   fragment_clusters = coherence_single_lag(coherence_setup, lag_idx=0)
   selected_clusters = supercluster_single_lag(
       supercluster_setup,
       config,
       fragment_clusters,
       lag_idx=0,
       xtalk=xtalk,
       td_inputs_cache=td_inputs_cache,
   )
   # None means that no supercluster survived for this lag.
   clusters = [] if selected_clusters is None else selected_clusters.clusters

Finally, calculate likelihood statistics for accepted clusters:

.. code-block:: python

   accepted = []
   for cluster_id, cluster in enumerate(clusters, start=1):
       if cluster.cluster_status > 0:
           continue
       result_cluster, sky_stats = evaluate_cluster_likelihood(
           config.nIFO,
           cluster,
           config,
           cluster_id=cluster_id,
           nRMS=nRMS,
           setup=likelihood_setup,
           xtalk=xtalk,
       )
       if result_cluster is not None and result_cluster.cluster_status == -1:
           accepted.append((result_cluster, sky_stats))

Matching cWB target-SNR resampling
----------------------------------

The native segment processor defaults to cWB-compatible resampling for
target-SNR injections (cWB simulation modes 2 or 5):

.. code-block:: yaml

   injection_resampling: cwb

The target-SNR multiplier is estimated with FFT resampling. In this mode,
the estimator uses the unmodified noise RMS before the conditioning bandpass,
then reproduces ``detector::setsim``: two energy-centroid iterations, a
99.9-percent energy crop, and the reference packed-FFT frequency cut.
This convention matters even when the generated waveform is identical. The final scaled
injection is then downsampled with Meyer(1024), separately from the noise,
before conditioning. Fixed-hrss trials retain FFT resampling. A trial containing
both target-SNR and fixed-hrss sources is rejected with this option; schedule
them as separate trials. Set ``injection_resampling: fft`` explicitly to retain the previous FFT-only behavior.
This option does not select the SNR population: source parameters must still
provide ``target_snr`` (or ``targeted_snr``).

Waveform generation and SNR scaling
-----------------------------------

Use ``burst_waveform.interface.generate_waveform_pycwb.get_td_waveform`` for
cWB SG, SGE, GA, and WNB waveforms. The duplicate
``pycwb.modules.injection.burst_population`` module has been removed; install
``burst-waveform>=0.5.0`` when migrating.
SGE ``hrss`` is the source amplitude before inclination weighting; ``iota``
is in radians (``ellipticity`` remains an alias). WNB does not receive
inclination factors. Its ``duration`` is the Gaussian amplitude standard
deviation, and ``frequency`` is the lower band edge. WNB defaults to cWB's
``mode: 0``; specify ``mode: 1`` to retain the symmetric construction formerly
used unconditionally by ``burst_population``. NumPy and ROOT random streams
differ, so equal seeds do not imply equal WNB realizations.

``pycwb.modules.injection.snr_scaling.target_snr_scales`` computes per-source
network-SNR multipliers from clean detector data.
Population parameter generation remains separate from signal normalization.
Meyer resampling lives in ``pycwb.modules.data_conditioning.resampling`` and
is applied by the segment workflow before regression and whitening.

The complete ``pycwb run`` path also saves triggers, reconstructed waveforms,
injection products, Q-veto values, plots, and catalog rows according to the
output options in the YAML file.


----

You have learned
----------------

- ✅ How to configure waveform injections in ``user_parameters.yaml``
- ✅ How to run an injection search with ``pycwb run``
- ✅ How to inspect the data conditioning pipeline step by step
- ✅ How likelihood evaluation and cluster acceptance work

**Try the workflow:** :doc:`tutorial_population` demonstrates recovery and
matching for several sources with different amplitudes.
