.. _catalog_format:

Catalog format and provenance
=============================

For trigger fields, units, and the output directory layout, start with
:doc:`understanding_results`. The complete current trigger schema is documented
by :py:class:`pycwb.types.trigger.Trigger`.

Catalog job provenance
----------------------

New master catalogs keep job descriptions in an immutable ``jobs.parquet``
next to ``catalog.parquet``. The catalog footer contains a versioned
``pycwb_jobs_manifest`` reference with a relative path and manifest identity.
The reader checks that identity when loading jobs. Older catalogs and batch
fragments with inline ``jobs`` metadata remain supported. Ordinary Parquet
inputs with neither inline jobs nor an explicit reference have no job metadata;
they do not implicitly use a nearby ``jobs.parquet``.

Selection, simulation matching and filtering, and scoring preserve source
provenance in their single-run trigger outputs. A selected or scored catalog in
another directory references the original manifest through a rebased relative
path. It does not copy the full job list into each output. Different runs can
therefore write outputs into the same temporary directory without sharing job
metadata accidentally.

The manifest describes the full source run. It does **not** describe selection
membership or selected exposure. Use the selection's job-ID, progress, interval,
and livetime outputs for those quantities; jobs with no triggers may still
contribute exposure. Multi-run combined tables retain their separate per-run
provenance model.

When transferring results, include the referenced manifest and preserve the
relative directory layout. Moving only a manifest-backed catalog is insufficient.
A missing, corrupt, or mismatched explicit manifest is an error, including when
building a report. Do not remove its reference to suppress the error.

Pre-release files produced with an unmarked sibling manifest must be regenerated
from their run inputs before using this format. The reader deliberately does not
guess whether an unmarked Parquet file was intended to depend on that manifest.
