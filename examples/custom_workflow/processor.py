"""A custom segment processor selected by YAML and run by ``pycwb run``.

This recipe adds candidate diagnostics between superclustering and likelihood.
It keeps native preparation and persistence and runs pending lags serially.
"""

import logging

from pycwb.modules.super_cluster_native.super_cluster import supercluster_single_lag
from pycwb.workflow.subflow import process_job_segment_native as native

logger = logging.getLogger(__name__)


def report_candidates(fragment_cluster, lag):
    """Observe the candidates without changing selections or cluster contents."""
    count = 0 if fragment_cluster is None else sum(
        cluster.cluster_status <= 0 for cluster in fragment_cluster.clusters
    )
    logger.info("Custom workflow: lag %d has %d candidates before likelihood", lag, count)


def supercluster_with_diagnostics(setup, config, fragments, lag, **kwargs):
    """Compose two ordinary calls; no stage registration is required."""
    fragment_cluster = supercluster_single_lag(setup, config, fragments, lag, **kwargs)
    report_candidates(fragment_cluster, lag)
    return fragment_cluster


def process_lags(context, output_context, skip_lags):
    """Own the lag loop while reusing the supplied recipe's analysis and output."""
    # These private helpers are reused deliberately for this built-in recipe.
    # A different scientific sequence can call the modules directly here.
    for lag in native._iter_pending_lags(context, skip_lags):
        result = native._run_lag_analysis(
            context, lag, supercluster=supercluster_with_diagnostics,
        )
        native._save_lag_outputs(output_context, result)


def process_job_segment(working_dir, config, job_seg, **kwargs):
    """CLI entry point: reuse native preparation and choose our own lag loop."""
    return native.process_job_segment(
        working_dir, config, job_seg, **kwargs, lag_processor=process_lags,
    )


# The wrapper forwards the input provider to native preparation unchanged.
process_job_segment.supports_input_provider = True
