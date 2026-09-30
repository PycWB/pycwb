.. _recipe_injections:

Injection Campaign
==================

**Use this route for:** testing recovery over your chosen source population.
Supply a base noise/data configuration, waveform generator, source parameters
and the amplitude, sky and time distributions you want to study.

1. Follow :doc:`injection_infrastructure` for parameter lists or Python-generated
   populations, scheduling, repeated trials and generated or real detector noise.
2. Check waveform support and segment eligibility. Fixed source amplitude and
   target network SNR define different populations; use the conventions in
   :doc:`tutorial_injection` when selecting normalization.
3. Run locally first, then follow :ref:`cluster_injection_campaigns` to scale the
   same configuration. Parallel execution settings are in :doc:`workflow_execution`.
4. Build simulation truth and match it to recovered triggers with
   :doc:`postproduction_trainingset`; use :doc:`postproduction_study` for efficiency.

**Completion check:** every scheduled source has a traceable identity and
eligibility record, including sources with no recovered trigger. Inspect timing,
amplitude and matching before interpreting a recovery fraction.

**Worked example:** :doc:`tutorial_population`; for a custom generator,
:doc:`tutorial_customized_wf_gen`.

See :doc:`analysis_recipes` for the other analysis tasks.
