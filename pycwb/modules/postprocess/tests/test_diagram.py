"""Tests for workflow DAG diagram generation.

Tests the ``@action_spec`` decorator, DAG builder, Mermaid/DOT renderers,
and integration with the full O4_K21 workflow.

Run with::

    pytest pycwb/modules/postprocess/tests/test_diagram.py -v
"""

from __future__ import annotations

import os
import tempfile
import textwrap
import unittest

import yaml

from pycwb.post_production.action_spec import action_spec, get_action_spec
from pycwb.post_production.diagram import (
    build_dag,
    generate_workflow_diagram,
    render_dot,
    render_mermaid,
)


# ---------------------------------------------------------------------------
# Minimal decorated functions for testing
# ---------------------------------------------------------------------------

@action_spec(
    outputs=['output_file'],
    inputs=['progress_file'],
    description='Select jobs by live time',
)
def _test_select_jobs(work_dir, progress_file, output_file, fraction=0.5, **kwargs) -> dict:
    return {}


@action_spec(
    outputs=['output_file'],
    inputs=['input_file', 'job_ids_file'],
    description='Filter catalog by jobs',
)
def _test_filter_catalog(work_dir, input_file, job_ids_file, output_file, **kwargs) -> dict:
    return {}


@action_spec(
    outputs=['model_file'],
    inputs=['bkg_catalog', 'sim_catalog'],
    description='Train XGBoost',
)
def _test_train(work_dir, bkg_catalog, sim_catalog, model_file, **kwargs) -> dict:
    return {}


# Undecorated function (fallback test)
def _test_undecorated(work_dir, some_input, some_output, **kwargs) -> dict:
    """Undecorated function docstring."""
    return {}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestActionSpec(unittest.TestCase):
    """Tests for the @action_spec decorator and get_action_spec helper."""

    def test_decorator_stores_metadata(self):
        """@action_spec stores inputs, outputs, description on __action_spec__."""
        spec = _test_select_jobs.__action_spec__
        self.assertEqual(spec['outputs'], ['output_file'])
        self.assertEqual(spec['inputs'], ['progress_file'])
        self.assertIn('Select jobs', spec['description'])

    def test_get_action_spec_decorated(self):
        """get_action_spec returns metadata for decorated functions."""
        spec = get_action_spec(_test_select_jobs)
        self.assertEqual(spec['outputs'], ['output_file'])

    def test_get_action_spec_undecorated(self):
        """get_action_spec returns empty spec for undecorated functions."""
        spec = get_action_spec(_test_undecorated)
        self.assertEqual(spec['outputs'], [])
        self.assertEqual(spec['inputs'], [])

    def test_get_action_spec_fallback_description(self):
        """get_action_spec uses docstring as description fallback."""
        spec = get_action_spec(_test_undecorated)
        self.assertEqual(spec['description'], 'Undecorated function docstring.')


class TestBuildDag(unittest.TestCase):
    """Tests for the build_dag function."""

    def _make_workflow(self, steps):
        return {
            'global': {'work_dir': '/tmp/test', 'search': 'blf', 'nifo': 2},
            'steps': steps,
        }

    def test_empty_workflow(self):
        """Empty step list produces nodes-only DAG."""
        dag = build_dag(self._make_workflow([]))
        self.assertEqual(len(dag['nodes']), 0)
        self.assertEqual(len(dag['edges']), 0)

    def test_single_step_no_deps(self):
        """Single step with no matching inputs produces 1 node, 0 edges."""
        dag = build_dag(self._make_workflow([{
            'action': 'pycwb.modules.postprocess.tests.test_diagram._test_select_jobs',
            'args': {
                'progress_file': 'data/progress.parquet',
                'output_file': 'data/jobs.txt',
            },
        }]))
        self.assertEqual(len(dag['nodes']), 1)
        self.assertEqual(len(dag['edges']), 0)

    def test_file_path_matching_creates_edge(self):
        """When step 1 output_file == step 2 input arg, an edge is created."""
        dag = build_dag(self._make_workflow([
            {
                'action': 'pycwb.modules.postprocess.tests.test_diagram._test_select_jobs',
                'args': {
                    'progress_file': 'data/progress.parquet',
                    'output_file': 'data/jobs.txt',
                },
            },
            {
                'action': 'pycwb.modules.postprocess.tests.test_diagram._test_filter_catalog',
                'args': {
                    'input_file': 'data/catalog.parquet',
                    'job_ids_file': 'data/jobs.txt',  # ← matches step 0 output_file
                    'output_file': 'data/filtered.parquet',
                },
            },
        ]))
        self.assertEqual(len(dag['nodes']), 2)
        self.assertEqual(len(dag['edges']), 1)
        edge = dag['edges'][0]
        self.assertEqual(edge['from'], 'step0')
        self.assertEqual(edge['to'], 'step1')
        self.assertEqual(edge['via'], 'file')

    def test_alias_matching_creates_edge(self):
        """When a target arg value matches a source output_alias, edge created."""
        dag = build_dag(self._make_workflow([
            {
                'action': 'pycwb.modules.postprocess.tests.test_diagram._test_select_jobs',
                'args': {
                    'progress_file': 'data/progress.parquet',
                    'output_file': 'data/jobs.txt',
                },
                'output_alias': 'my_jobs',
            },
            {
                'action': 'pycwb.modules.postprocess.tests.test_diagram._test_undecorated',
                'args': {
                    'some_input': 'my_jobs',  # ← matches alias
                    'some_output': 'out.txt',
                },
            },
        ]))
        self.assertEqual(len(dag['edges']), 1)
        edge = dag['edges'][0]
        self.assertEqual(edge['from'], 'step0')
        self.assertEqual(edge['to'], 'step1')
        self.assertEqual(edge['via'], 'alias')

    def test_no_false_match(self):
        """Different file paths do NOT create an edge."""
        dag = build_dag(self._make_workflow([
            {
                'action': 'pycwb.modules.postprocess.tests.test_diagram._test_select_jobs',
                'args': {
                    'progress_file': 'data/progress.parquet',
                    'output_file': 'data/jobs_train.txt',
                },
            },
            {
                'action': 'pycwb.modules.postprocess.tests.test_diagram._test_filter_catalog',
                'args': {
                    'input_file': 'data/catalog.parquet',
                    'job_ids_file': 'data/jobs_farrho.txt',  # ≠ step 0 output_file
                    'output_file': 'data/filtered.parquet',
                },
            },
        ]))
        self.assertEqual(len(dag['edges']), 0)

    def test_multi_step_chain(self):
        """A three-step chain (A→B→C) produces 2 edges."""
        dag = build_dag(self._make_workflow([
            {
                'action': 'pycwb.modules.postprocess.tests.test_diagram._test_select_jobs',
                'args': {
                    'progress_file': 'data/progress.parquet',
                    'output_file': 'data/jobs.txt',
                },
            },
            {
                'action': 'pycwb.modules.postprocess.tests.test_diagram._test_filter_catalog',
                'args': {
                    'input_file': 'data/catalog.parquet',
                    'job_ids_file': 'data/jobs.txt',
                    'output_file': 'data/filtered.parquet',
                },
            },
            {
                'action': 'pycwb.modules.postprocess.tests.test_diagram._test_train',
                'args': {
                    'bkg_catalog': 'data/filtered.parquet',  # ← matches step 1 output_file
                    'sim_catalog': 'data/sim.parquet',
                    'model_file': 'models/xgb.ubj',
                },
            },
        ]))
        self.assertEqual(len(dag['edges']), 2)


class TestRenderers(unittest.TestCase):
    """Tests for Mermaid and DOT renderers."""

    def setUp(self):
        self.dag = {
            'nodes': [
                {
                    'id': 'step0', 'label': 'select_jobs_by_livetime',
                    'description': 'Select jobs', 'action': 'a.b.c',
                    'step_index': 0, 'output_alias': None, 'args': {},
                },
                {
                    'id': 'step1', 'label': 'filter_catalog_by_jobs',
                    'description': 'Filter catalog', 'action': 'a.b.d',
                    'step_index': 1, 'output_alias': None, 'args': {},
                },
            ],
            'edges': [
                {'from': 'step0', 'to': 'step1', 'label': 'jobs.txt', 'via': 'file'},
            ],
        }

    def test_mermaid_output_structure(self):
        """Mermaid output contains key elements."""
        mmd = render_mermaid(self.dag, 'Test')
        self.assertIn('graph TD', mmd)
        self.assertIn('step0', mmd)
        self.assertIn('step1', mmd)
        self.assertIn('```mermaid', mmd)
        self.assertIn('title: Test', mmd)

    def test_mermaid_has_edges(self):
        """Mermaid output includes edge markup."""
        mmd = render_mermaid(self.dag, 'Test')
        self.assertIn('-->|', mmd)
        self.assertIn('step0', mmd)
        self.assertIn('step1', mmd)

    def test_dot_output_structure(self):
        """DOT output contains key elements."""
        dot = render_dot(self.dag, 'Test')
        self.assertIn('digraph workflow', dot)
        self.assertIn('step0', dot)
        self.assertIn('step1', dot)
        self.assertIn('->', dot)

    def test_empty_dag_renders(self):
        """Empty DAG renders without errors."""
        empty = {'nodes': [], 'edges': []}
        mmd = render_mermaid(empty)
        dot = render_dot(empty)
        self.assertIn('graph TD', mmd)
        self.assertIn('digraph workflow', dot)


class TestFullWorkflow(unittest.TestCase):
    """Integration test: generate diagram from the full O4_K21 workflow."""

    @classmethod
    def setUpClass(cls):
        cls.workflow_path = os.path.join(
            os.path.dirname(__file__), 'O4_K21', 'postprocess_workflow_std.yaml',
        )

    def test_full_workflow_dag_builds(self):
        """DAG can be built from the full O4_K21 workflow without errors."""
        if not os.path.exists(self.workflow_path):
            self.skipTest(f"Workflow file not found: {self.workflow_path}")

        with open(self.workflow_path, 'r') as f:
            wf = yaml.safe_load(f)

        dag = build_dag(wf)
        n_steps = len(wf['steps'])
        self.assertEqual(len(dag['nodes']), n_steps,
                         f"Expected {n_steps} nodes, got {len(dag['nodes'])}")

        # There should be many edges in a real workflow
        self.assertGreater(len(dag['edges']), 3,
                           f"Expected >3 edges in full workflow, got {len(dag['edges'])}")

    def test_full_workflow_mermaid_renders(self):
        """Mermaid can be rendered from the full workflow."""
        if not os.path.exists(self.workflow_path):
            self.skipTest(f"Workflow file not found: {self.workflow_path}")

        with open(self.workflow_path, 'r') as f:
            wf = yaml.safe_load(f)

        dag = build_dag(wf)
        mmd = render_mermaid(dag, 'O4_K21 Workflow')
        self.assertIn('graph TD', mmd)
        # Should have all step nodes
        for i in range(len(wf['steps'])):
            self.assertIn(f'step{i}', mmd)

    def test_full_workflow_dot_renders(self):
        """DOT can be rendered from the full workflow."""
        if not os.path.exists(self.workflow_path):
            self.skipTest(f"Workflow file not found: {self.workflow_path}")

        with open(self.workflow_path, 'r') as f:
            wf = yaml.safe_load(f)

        dag = build_dag(wf)
        dot = render_dot(dag, 'O4_K21 Workflow')
        self.assertIn('digraph workflow', dot)
        for i in range(len(wf['steps'])):
            self.assertIn(f'step{i}', dot)

    def test_generate_workflow_diagram_writes_files(self):
        """generate_workflow_diagram writes .mmd and .dot files."""
        if not os.path.exists(self.workflow_path):
            self.skipTest(f"Workflow file not found: {self.workflow_path}")

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = os.path.join(tmpdir, 'test_diagram')
            result = generate_workflow_diagram(
                self.workflow_path,
                output_prefix=prefix,
                generate_png=False,  # skip PNG to avoid graphviz dependency
            )

            # .mmd must exist
            self.assertTrue(os.path.exists(result['mmd']),
                            f"Missing .mmd: {result['mmd']}")
            with open(result['mmd'], 'r') as f:
                content = f.read()
            self.assertIn('graph TD', content)

            # .dot must exist
            self.assertTrue(os.path.exists(result['dot']),
                            f"Missing .dot: {result['dot']}")
            with open(result['dot'], 'r') as f:
                content = f.read()
            self.assertIn('digraph workflow', content)

            # .png should be None (skipped)
            self.assertIsNone(result['png'])

    def test_redesign_workflow_dag_builds(self):
        """DAG can be built from the redesigned O4_K21 workflow."""
        workflow_path = os.path.join(
            os.path.dirname(__file__), 'O4_K21', 'postprocess_workflow_redesign_test.yaml',
        )
        if not os.path.exists(workflow_path):
            self.skipTest(f"Workflow file not found: {workflow_path}")

        with open(workflow_path, 'r') as f:
            wf = yaml.safe_load(f)

        dag = build_dag(wf)
        self.assertEqual(len(dag['nodes']), len(wf['steps']))
        self.assertGreaterEqual(len(dag['edges']), 10)


if __name__ == '__main__':
    unittest.main()
