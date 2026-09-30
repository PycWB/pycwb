# Documentation structure and preservation record

Date: 2026-09-29. Scope: documentation organization, navigation, and presentation.
Scientific algorithms, configuration values, example commands, and numerical
claims were not revised in this change.

## Primary navigation

| Section | Landing page | Primary responsibility |
| --- | --- | --- |
| Getting started | `source/getting_started.rst` | Installation, synthetic first search, interpreting its output |
| Tutorials | `source/tutorials.rst` | Guided injection and batch examples, with prerequisites and outcomes |
| Run analyses | `source/run_analyses.rst` | Task recipes, production setup, targeting, injection configuration, postproduction, troubleshooting, reproducibility |
| Concepts and methods | `source/core_concepts.rst` | Pipeline, intermediate Python walkthrough, job control, reconstruction, significance and sensitivity |
| Reference | `source/reference.rst` | YAML, CLI, execution/backends, catalogs, scientific conventions, validation scope, action interfaces, Python API |
| Development | `source/development.rst` | Setup, architecture, module development, postproduction internals, performance, testing, contributions, releases, C++ maintenance |
| About PycWB | `source/about.rst` | Scientific introduction, original visuals and badges, citation, heritage, public examples, compatibility, community |

Each document has at most one parent in the Sphinx navigation tree. Cross-links
connect workflows to their explanations and reference pages. The sidebar expands
only the active branch. Full software version information remains in browser
page titles, the sidebar, and the homepage; the site header uses the short title.

## Content relocation

| Original location | Destination / treatment |
| --- | --- |
| Homepage introduction, “What is pycWB?”, animation and caption, badges | `about.rst`; original prose, asset references and caption retained verbatim |
| Homepage quick-start commands and source-checkout prerequisite | `getting_started.rst`; complete block retained verbatim; detailed first tutorial unchanged |
| Homepage three-command CLI summary | `cli_reference.rst`; command block retained verbatim |
| Homepage ten cards and documentation map | Six main sections and About; all destination topics remain available |
| Homepage indexes/search links | `reference.rst#reference-indexes`, linked from the old homepage anchor |
| Analysis Recipes: All-Sky Short Burst Search | `recipe_all_sky.rst` |
| Analysis Recipes: Targeted External-Trigger Search | `recipe_targeted.rst` |
| Analysis Recipes: Injection Campaign | `recipe_injections.rst` |
| Analysis Recipes: Background-Only Production | `recipe_background.rst` |
| Analysis Recipes: Training XGBoost Ranking | `recipe_training.rst` |
| Analysis Recipes: Efficiency Study | `recipe_efficiency.rst` |
| Analysis Recipes: Debugging a Failed Production | `recipe_debugging.rst` |
| Postproduction architecture and module table | `dev_postproduction.rst`; body retained verbatim |
| Postproduction catalog job provenance | `catalog_format.rst`; body retained verbatim, including compatibility, exposure and transfer caveats |
| Release-policy maintainer checklist | `dev_release.rst`; checklist and publishing caveat retained verbatim |
| Learning Path repeated diagram, lesson list and navigation | One tutorial table preserving lesson destinations, outcomes, order and approximate times |
| Beginner description of the Python search walkthrough | Corrected to intermediate, matching the existing destination's first-search prerequisite |
| Ambiguous star ratings in the learning path | Replaced with explicit prerequisites; no unverified new difficulty or runtime claims |
| `choose_your_path.rst` | Retained as a secondary audience/task directory, linked from About; all audience routes retained |
| `decision_guides.rst` | Retained with its existing placeholder warning and illustrations; linked from Choose Your Path, not promoted as validated guidance |
| `package.rst` | Retained as package shortcuts, linked from Reference; generated API tree has one canonical parent |

Every extracted recipe body retains its original nonblank lines, including YAML, shell commands,
inputs, expected outputs and checks. Blank lines were added after list introductions
so Sphinx renders their bullets correctly. The recipe index preserves its old section
anchors and links each to the corresponding new page. The original postproduction
and release-checklist anchors likewise remain and point readers to the new pages.
No original source page was deleted. Old homepage section anchors also remain.

## Preservation checks

Compared against a source snapshot and rendered anchor inventory taken before
editing this documentation:

- All **115 original root-level RST source pages** remain.
- All **195 original code blocks** across **56 authored pages** are still present
  verbatim somewhere in the documentation source.
- All **15 extracted content blocks** retain every nonblank line in their recorded
  destinations (seven recipes, three homepage introduction/asset blocks, quick
  start, CLI summary, postproduction architecture, provenance, release checklist).
- All **824 original non-code document anchors** from those authored pages
  remain in their original compiled pages. Layout containers and generated
  code-line IDs are excluded from this comparison.
- **20,860 local document links** were checked against compiled files and anchor
  IDs: no missing destinations. External URLs, the browser's special `#top`
  fragment, and the theme's global skip-to-content links are excluded.
- Recipe list spacing was repaired during visual inspection; code blocks are unchanged.
- No duplicate non-code IDs were found in the compiled root-level HTML pages.
- No duplicate toctree parents: 127 documents belong to the main tree; the three
  secondary pages (`choose_your_path`, `decision_guides`, `package`) remain
  searchable and are explicitly linked from the main tree.
- The Sphinx HTML build passes with warnings treated as errors. One existing
  short heading underline in `tutorial_injection.rst` was repaired without
  changing its text.
- The final build reread every source and rewrote every page (`-E -a`). An
  intermediate incremental build emitted an import-order warning for
  `pycwb.workflow.batch`; the final full build passed without warnings, and no
  Python implementation was changed for this refactor.
- `git diff --check` passes.

Build command (using an environment containing `docs/requirements.txt`):

```sh
PYCWB_DOCS_OFFLINE=1 python -m sphinx -E -a -b html -W --keep-going docs/source docs/build/html
```

These checks establish structural preservation and link integrity. They do not
revalidate scientific recommendations, execute examples, or check remote sites.
Generated Python API documentation follows the current checkout's code and is
not covered by the authored-page anchor preservation claim.

## Rules for future documentation edits

1. Give each topic a single canonical navigation parent; use cross-links elsewhere.
2. Keep the homepage focused on entry points, with detailed content in topic pages.
3. Move substantive content before compacting its old location; retain a clear link.
4. Preserve old filenames and section anchors when relocating published content.
5. Preserve example inputs, commands, units, assumptions and limitations together.
6. Keep draft or experimental status explicit, and avoid promoting placeholders as guidance.
7. Build with warnings as errors and check document links after moving pages.

## Installation-page follow-up

The user's subsequent installation review explicitly replaces some of the
original prose and commands; the preservation totals above describe the initial
structural refactor, before these requested editorial changes.

- Recommend Python 3.13 while retaining the Python >=3.11 requirement.
- State that ROOT is not required for PycWB >=1 and direct ROOT-based analyses
  to original cWB. Remove the prior PyROOT installation recommendation.
- Explain that XGBoost is needed for model training/scoring in standard
  postproduction; show extras installed from the same source checkout.
- Move stable/prerelease instructions to Releases and Compatibility, retaining
  their commands and cautions about matching software and documentation.
- Move the platform/CI table to Validation Scope and Limitations.
- Move the documentation-build instructions to Build & Test.
- Keep the old installation section anchors as links to the relocated material.

## Remote synchronization and CLI cleanup

Fast-forwarded `integration/fix-injection-release` from `38ac0cb` to `2716b8d`
before the CLI cleanup. The remote regression target/witness guidance in
`backends.rst` was retained under Reference. Local tracked edits were compared
before and after the pull and were unchanged byte-for-byte.

At the user's request, removed the redundant environment-inventory subcommand
and its dedicated tests, registry entry, type-check target, and documentation
references. Environment recording now uses standard Python/pip commands. This
intentional removal supersedes the initial preservation totals for that command
and its generated reference anchors. A fresh documentation output tree replaced
the previous preview, removing stale generated API pages and search entries.

Validation: 32 onboarding, runtime-validation, and frame-reader module tests
passed; the quality checker reported no new debt; CLI help/version succeeded;
the removed command was rejected; a clean strict Sphinx build passed; local
document links resolved.
