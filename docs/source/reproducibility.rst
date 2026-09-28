.. _reproducibility:

Reproducing and Archiving an Analysis
=====================================

A fixed seed is one part of reproducibility. Preserve the software, inputs,
resolved settings and selection that produced the result.

Record the environment
----------------------

Run these in the analysis environment and save them beside the run:

.. code-block:: bash

   pycwb --version > pycwb-version.txt
   pycwb doctor --json > environment-report.json
   python -m pip freeze > requirements-analysis.txt
   conda env export > environment-analysis.yml

For source installs, record ``git rev-parse HEAD`` and any local patch. Also
record the revision of ``pycwb-config`` and custom waveform or conditioning
modules. A package version does not identify uncommitted source changes.

Preserve inputs and decisions
-----------------------------

* Keep the original YAML, its rendered/resolved configuration, workflow YAML,
  machine profile, and exact CLI overrides.
* Record frame identifiers, detector channels, data-quality segments and their
  sources. Preserve copies or immutable retrieval identifiers.
* Record the cross-talk catalog filename and SHA-256 checksum, plus any PSDs,
  sky maps, model files and custom input tables.
* Keep random seeds, injection populations, job/trial/lag selections, train/FAR
  splits and selected analyzed exposure.
* Record CPU/GPU backend, thread counts, and the YAML ``execution_profile``
  and ``gpu`` settings saved with the catalog. Runtime choices are explicit
  configuration, not ``PYCWB_*`` environment overrides.
  Numerical results can vary across dependency versions and hardware.

For example, compute a file checksum without loading it all into memory:

.. code-block:: python

   import hashlib
   from pathlib import Path

   path = Path("wdmXTalk/OverlapCatalog16-1024.bin")
   digest = hashlib.sha256()
   with path.open("rb") as stream:
       for block in iter(lambda: stream.read(1024 * 1024), b""):
           digest.update(block)
   print(path.name, digest.hexdigest())

Archive a complete run
----------------------

Master catalogs reference an immutable ``jobs.parquet`` manifest. Copy the
catalog, progress files, referenced manifest, configuration, model artifacts and
logs together, retaining relative paths. Filtered catalogs can reference a
manifest outside their own directory. Resolve it before archiving.

.. code-block:: python

   from pycwb.modules.catalog import Catalog

   catalog = Catalog.open("my_first_search/catalog/catalog.parquet")
   print(len(catalog.jobs))  # verifies the referenced manifest identity

The job manifest describes the source jobs, not the subset of exposure selected
by later postproduction. Preserve selection intervals and livetime products too.
See :ref:`postproduction` for the catalog provenance contract.

Re-run and compare
------------------

Restore the recorded environment and inputs in a new working directory.
Compare job completion and selected exposure before comparing trigger counts,
waveforms, statistics or efficiency. Use tolerances stated by the relevant
validation test; bitwise identity is not guaranteed across platforms.

Before sharing environment exports or logs publicly, remove credentials,
private data locations and access tokens. Citation guidance is in :ref:`credits`.
