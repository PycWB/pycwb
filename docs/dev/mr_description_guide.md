# MR Description Guide

This file is the single source of truth for writing Merge Request descriptions in pycWB.
It also contains the AI prompt templates — attach this file as context when asking an AI to draft a description.

---

## Why good MR descriptions matter

- Reviewers understand *what* changed and *why* without reading every diff line.
- Changelogs and release notes can be generated directly from them.
- Future contributors can trace a bug fix or feature back to its motivation.

---

## Keep MRs small

**One logical change per MR.**

Large MRs are harder to review, harder to revert, and harder to bisect. Rules of thumb:

- One bug fix per MR — if you find a second bug while fixing the first, open a separate branch.
- One feature per MR — split preparatory refactors from the feature itself.
- Rename/restructure separately from behaviour changes; mixed diffs obscure intent.

If a MR genuinely cannot be split, say so explicitly in the description and explain why.

---

## Branch and commit hygiene

### Rebase before opening the MR

Rebase onto the current `main` so the diff is clean and the commit history is coherent:

```bash
git fetch origin
git rebase origin/main
# resolve any conflicts, then:
git rebase --continue
```

### Force-push after rebase

Rebase rewrites history, so force-push your branch:

```bash
git push --force-with-lease origin <your-branch>
```

`--force-with-lease` is safer than `--force`: it aborts if someone else has pushed to the branch since your last fetch.

### Squash noise commits

Squash commits like "fix typo", "wip", or "try again" into meaningful commits before review:

```bash
git rebase -i origin/main   # mark noise commits as 'squash' or 'fixup'
git push --force-with-lease origin <your-branch>
```

---

## How to generate the MR description with AI

### Recommended: agent mode (one step)

1. Open GitHub Copilot Chat in VS Code and switch to **Agent mode**.
2. Attach this file (`docs/dev/mr_description_guide.md`) as context.
3. Type:

   ```
   Write an MR description for the current branch to main.
   ```

The agent will run `git diff` against `main` automatically, read this guide, generate the description, and **write it to `MR_DESCRIPTION.md`** in the repository root. Open that file to review and copy the content into the GitLab/GitHub MR form.

> **Tip — mark fixes in code comments before asking.** Add a short inline comment at each change site explaining *why* the change was made. The agent reads the diff and lifts those comments into the description automatically — and they stay in the code as permanent documentation for future contributors.
>
> Example:
> ```python
> # ROOT fills products for j in [K, n-K] then trims edge_samples from the full
> # array, so the effective trim on the product sub-array is (edge_samples - K).
> edge_v = max(0, edge_samples - K)
> ```

### Fallback: standard chat mode

If agent mode is unavailable, run this command and paste the output into the chat along with this guide file:

```bash
git diff origin/main...HEAD
```

For a large diff, get the file summary first, then diff files individually:

```bash
git diff --stat origin/main...HEAD
git diff origin/main...HEAD -- path/to/file.py
```

Then ask: *"Write an MR description for these changes following the rules in the attached guide, and write the result to MR_DESCRIPTION.md."*

### Review before posting

The AI output is a first draft:

- Verify every factual claim (file names, function names, numbers).
- Remove anything the AI hallucinated.
- Check the title fits the web UI title field (plain text, ≤ 72 characters, no backticks or markdown).

---

## Instructions for the AI agent

> This section is addressed to the AI. Follow it when asked to write an MR description.

1. **Get the diff.** Run `git diff origin/main...HEAD` to discover all changed files and their content. If the output exceeds your context limit, run `git diff --stat origin/main...HEAD` first, then diff each file individually.
2. **Determine the MR type.** Infer from the diff whether this is a bug fix or a new feature. If unclear, ask the user.
3. **Read inline comments.** Comments added at change sites explain the *why*. Use them as the primary source for the Problems and Changes sections — do not paraphrase away their meaning.
4. **Apply the structure.** Use the matching template from the "Required structure and formatting rules" section below. Every section is required; write "None." if it does not apply.
5. **Write the result to `MR_DESCRIPTION.md`.** Overwrite the file in the repository root with the generated description. Do not print the description to the chat — the file is the deliverable. The content must start directly with the plain-text title; no preamble, no explanation, no extra markdown wrapper.

---

## Posting the MR

1. Push your branch and open a new MR/PR on GitLab or GitHub.
2. **Title field**: plain-text title only — no backticks, no markdown formatting.
3. **Description field**: paste the markdown body starting from `## Summary`. Both GitLab and GitHub render it.
4. Assign reviewers, set labels (`bugfix`, `feature`, etc.), and link related issues.

---

## Required structure and formatting rules

> The AI must follow these rules exactly when generating a description.

### Rules for all MR types

- **Title**: plain text, ≤ 72 characters, no backticks, no markdown.
- Every section listed below is required. Write "None." if a section does not apply.
- Use `##` for top-level sections and `###` for sub-sections within Problems or Changes.
- Use fenced code blocks (` ``` `) for all code and diff snippets.
- Use markdown tables for before/after comparisons and test file lists.
- Do not invent section names not listed here.

---

### Bug fix structure

```
Title
<plain text, ≤ 72 characters>

## Summary
One paragraph: what broke, how severely, and that it is now fixed.
Include the component name and the pipelines or workflows affected.

## Background
Context a reviewer needs: which subsystem, why the correctness matters,
and any relevant reference implementation (e.g. the C++/ROOT counterpart).

## Problems
### 1. <Short problem name>
Describe the correct/expected behaviour, what the code did instead,
and the measurable consequence (error magnitude, wrong sign, etc.).
Include a code snippet showing the incorrect code if helpful.
Repeat with ### 2., ### 3., etc. for each independent bug.

## Root Cause Confirmation
Diagnostic steps or test output that confirmed the root cause
and ruled out other explanations. Omit this section only if trivial.

## Changes
State which file(s) changed. For each, show the before/after code
with fenced code blocks and a one-line explanation of each change.

## Results
A markdown table with columns: Check | Before fix | After fix.
Rows should cover every intermediate quantity and the end-to-end outcome.

## Breaking Changes
"None." or an explicit list of removed/renamed public API, config keys,
or data format changes that require downstream action.

## Testing
A markdown table with columns: File | Purpose.
One row per test script used to verify the fix.
```

---

### Feature structure

```
Title
<plain text, ≤ 72 characters>

## Summary
One paragraph: what the feature does, why it was added,
and which workflows or modules it affects.

## Motivation
The concrete use case or limitation that this feature addresses.

## Design
Key design decisions, alternatives considered, and trade-offs accepted.
Include a diagram or pseudo-code if it aids understanding.

## Changes
List new files and modified files. For each, explain what it does
or what changed and why. Use fenced code blocks for relevant snippets.

## Breaking Changes
"None." or an explicit list with migration instructions.

## Testing
A markdown table with columns: File | Purpose.
```
