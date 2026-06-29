"""Tests for post-production workflow configuration helpers."""

from __future__ import annotations

import os

from pycwb.post_production.workflow_config import (
    prepare_step_args,
    resolve_value,
    store_result,
    workflow_base_context,
    workflow_runtime,
)


def test_vars_tmp_and_reference_resolution():
    workflow = {
        "global": {"work_dir": "/analysis"},
        "vars": {"paths": {"catalog": "BKG/catalog.parquet"}},
        "runtime": {"tmp_dir": "${work_dir}/scratch", "cleanup_tmp": "never"},
    }
    context = workflow_base_context(workflow)
    runtime = workflow_runtime(workflow, context)
    context["bkg_split"] = {
        "far": {
            "triggers_file": os.path.join(runtime["tmp_dir"], "bkg_far.parquet"),
            "livetime": {"seconds": 123.0},
        }
    }

    step = {
        "inputs": {
            "catalog_file": "${paths.catalog}",
            "far_catalog": "@bkg_split.far.triggers_file",
            "livetime": "@bkg_split.far.livetime.seconds",
        },
        "outputs": {"output_file": "tmp://out.parquet"},
    }
    args = prepare_step_args(step, context, runtime)

    assert args["catalog_file"] == "BKG/catalog.parquet"
    assert args["far_catalog"].endswith("scratch/bkg_far.parquet")
    assert args["livetime"] == 123.0
    assert args["output_file"].endswith("scratch/out.parquet")


def test_store_result_supports_id_and_list_aliases():
    context = {}
    store_result(context, {"id": "split"}, {"train": 1, "far": 2})
    assert context["split"] == {"train": 1, "far": 2}

    store_result(context, {"output_alias": ["a", "b"]}, [10, 20])
    assert context["a"] == 10
    assert context["b"] == 20


def test_resolve_value_expands_nested_values_without_refs():
    context = {"work_dir": "/w", "run": {"name": "O4"}}
    runtime = {"tmp_dir": "${work_dir}/tmp"}
    value = {"x": ["${run.name}", "tmp://a.txt", "@later.value"]}

    resolved = resolve_value(value, context, runtime, resolve_refs=False)

    assert resolved["x"][0] == "O4"
    assert resolved["x"][1] == "/w/tmp/a.txt"
    assert resolved["x"][2] == "@later.value"


def test_relative_runtime_tmp_dir_is_normalized_to_absolute_path():
    workflow = {
        "vars": {"work_dir": "tests/postprod/O4_K21"},
        "runtime": {"tmp_dir": "${work_dir}/tmp/redesign_test"},
    }
    context = workflow_base_context(workflow)
    runtime = workflow_runtime(workflow, context)

    assert os.path.isabs(runtime["tmp_dir"])
    assert runtime["tmp_dir"].endswith("tests/postprod/O4_K21/tmp/redesign_test")
