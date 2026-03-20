# AI Code Review Guide

This file is the single source of truth for automated code review in pycWB.
Attach it as context in VS Code Agent mode and type:

```
Review the changed files on the current branch against main.
```

The AI agent will run the diff, apply every check below, and report findings
grouped by category. It will also fix issues automatically when safe to do so.

---

## How the agent should operate

1. **Get the diff.** Run `git diff origin/main...HEAD --name-only` to list
   changed files, then read each changed file in full.
2. **Run every check** in the sections below, in order.
3. **Report** a numbered list of findings grouped by section heading. For each
   finding give the file path, line number, the problem, and the suggested fix.
4. **Apply safe fixes automatically** (docstring formatting, missing type
   hints, unused imports) unless the user says "report only".
5. After all fixes, re-run the relevant check tools to confirm zero remaining
   issues.

---

## 1. Docstrings — NumPy style

pycWB uses **NumPy-style docstrings** exclusively (numpydoc).

### What to check

- Every public module, class, method, and function **must** have a docstring.
- Private helpers (`_foo`) should have at least a one-liner if the logic is
  non-trivial.
- Docstrings must follow the [numpydoc format](https://numpydoc.readthedocs.io/en/latest/format.html):

```python
def compute_snr(strain, psd, flow=20.0):
    """
    Compute the matched-filter signal-to-noise ratio.

    Parameters
    ----------
    strain : np.ndarray
        Time-domain strain data.
    psd : np.ndarray
        One-sided power spectral density.
    flow : float, optional
        Low-frequency cutoff in Hz. Default is 20.0.

    Returns
    -------
    float
        The optimal SNR value.

    Raises
    ------
    ValueError
        If `strain` and `psd` have different lengths.

    Notes
    -----
    Uses the standard inner-product definition from [1]_.

    References
    ----------
    .. [1] Allen et al., FINDCHIRP, PRD 85, 122006 (2012).

    Examples
    --------
    >>> compute_snr(strain, psd)
    12.7
    """
```

### Section ordering

`Summary` → `Extended Summary` → `Parameters` → `Returns` / `Yields` →
`Raises` → `See Also` → `Notes` → `References` → `Examples`

### Common mistakes to flag

| Mistake | Fix |
|---|---|
| reStructuredText params (`:param x:`, `:type x:`) | Convert to `Parameters` section |
| Google-style (`Args:`, `Returns:`) | Convert to NumPy-style sections |
| Missing `Returns` section when function returns a value | Add section |
| Parameter not documented | Add entry |
| Documented parameter not in signature | Remove entry |
| `"""One-liner."""` on class/public function with args | Expand to full docstring |

### Tools the agent must use

| Step | Tool / Command | Why |
|---|---|---|
| Find missing docstrings | `grep_search` for `def ` and `class ` lines, then `read_file` to check next line | Locates undocumented symbols |
| Validate numpydoc format | `run_in_terminal`: `python -m pydocstyle --convention=numpy <file>` | Parses docstring structure; install with `pip install pydocstyle` if missing |
| Auto-fix format | `replace_string_in_file` / `multi_replace_string_in_file` | Convert reST/Google to NumPy style in-place |

---

## 2. Type annotations

### What to check

- All **public** function signatures must have type annotations for every
  parameter and the return type.
- Use **PEP 604 / PEP 585** modern syntax (`list[int]`, `dict[str, Any]`,
  `X | None`) with `from __future__ import annotations` at the top of the
  file when needed.
- Do **not** use the old `typing` wrappers (`List`, `Dict`, `Optional`,
  `Tuple`, `Union`) in new or modified code. If the file already imports them
  for unchanged code, leave those lines but use modern syntax for your changes.
- `*args` and `**kwargs` should be annotated when their types are known.
- Class attributes in dataclasses / attrs must be annotated.

### Common mistakes to flag

| Mistake | Fix |
|---|---|
| `Optional[X]` | `X \| None` |
| `List[int]` | `list[int]` |
| `Dict[str, Any]` | `dict[str, Any]` |
| `Tuple[int, ...]` | `tuple[int, ...]` |
| `Union[A, B]` | `A \| B` |
| Missing return type | Add `-> ReturnType` (use `-> None` for procedures) |
| Untyped parameter | Add annotation |

### Tools the agent must use

| Step | Tool / Command | Why |
|---|---|---|
| Scan for missing annotations | `run_in_terminal`: `python -m mypy --ignore-missing-imports --disallow-untyped-defs <file>` | Reports untyped defs; install with `pip install mypy` if missing |
| Quick grep for old typing imports | `grep_search` for `from typing import` in changed files | Finds files still using old-style generics |
| Auto-fix annotations | `replace_string_in_file` / `multi_replace_string_in_file` | Rewrite signatures in-place |

---

## 3. Code style & formatting

pycWB follows **PEP 8** with a maximum line length of **120 characters**.

### What to check

- Indentation: 4 spaces, no tabs.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes,
  `UPPER_SNAKE_CASE` for module-level constants.
- Imports: standard library → third-party → local, separated by blank lines.
  No wildcard imports (`from x import *`).
- No bare `except:` — always catch a specific exception.
- Use f-strings over `%` formatting or `.format()` for new code.
- String quotes: prefer double quotes `"` consistently (the dominant style in
  the codebase).
- Trailing whitespace and missing final newline.

### Tools the agent must use

| Step | Tool / Command | Why |
|---|---|---|
| Lint for PEP 8 violations | `run_in_terminal`: `python -m ruff check --line-length 120 <file>` | Fast linter; install with `pip install ruff` if missing |
| Auto-fix safe violations | `run_in_terminal`: `python -m ruff check --fix --line-length 120 <file>` | Ruff can auto-fix import sorting, unused imports, etc. |
| Format check | `run_in_terminal`: `python -m ruff format --check --line-length 120 <file>` | Checks formatting without modifying |
| Auto-format | `run_in_terminal`: `python -m ruff format --line-length 120 <file>` | Applies consistent formatting |
| Fallback if ruff unavailable | `run_in_terminal`: `python -m flake8 --max-line-length 120 <file>` | Install with `pip install flake8` |

> **Note:** Do not auto-format files that are not part of the current diff.
> Only format changed files.

---

## 4. Unused code

### What to check

- **Unused imports**: imported names that are never referenced in the file.
- **Unused variables**: assigned but never read (excluding `_` throwaway).
- **Unused function arguments**: parameters that are never used in the body
  (flag but do not auto-remove — they may be part of an interface contract).
- **Dead code**: unreachable branches (`if False:`, code after unconditional
  `return`/`raise`).
- **Commented-out code blocks**: more than 3 consecutive commented lines that
  look like code should be removed (use version control instead).

### Tools the agent must use

| Step | Tool / Command | Why |
|---|---|---|
| Detect unused imports & variables | `run_in_terminal`: `python -m ruff check --select F401,F841 <file>` | F401 = unused import, F841 = unused variable |
| Auto-remove unused imports | `run_in_terminal`: `python -m ruff check --fix --select F401 <file>` | Safe auto-fix |
| Detect dead / commented-out code | `grep_search` for patterns like `# .*=.*\(` or consecutive `#` lines, then `read_file` to inspect | Manual review needed |
| Detect unused function args | `run_in_terminal`: `python -m vulture <file>` | Reports unused code; install with `pip install vulture` if missing |

---

## 5. Comments

### What to check

- **Why, not what.** Comments should explain *why* something is done, not
  restate the code. Flag comments like `# increment i` next to `i += 1`.
- **Accuracy.** If the code changed but the comment did not, flag the stale
  comment.
- **TODOs.** Must include an owner or issue reference:
  `# TODO(username): fix edge case — see #123`. Flag bare `# TODO` or
  `# FIXME` with no context.
- **No commented-out code.** See §4 above.
- **Inline comments.** Place them on the same line only if short; otherwise
  put them on the preceding line.
- **Language.** Use English. Full sentences with a capital letter and period
  for multi-word comments.

### Tools the agent must use

| Step | Tool / Command | Why |
|---|---|---|
| Find TODOs/FIXMEs | `grep_search` for `TODO\|FIXME\|HACK\|XXX` in changed files | Surfaces action items |
| Find stale comments | `read_file` the changed hunks; compare comment text to surrounding code | Manual review — no automated tool |
| Find commented-out code | `grep_search` for `^\s*#\s*(def \|class \|import \|from \|return \|if \|for )` | Catches obvious commented-out statements |

---

## 6. Error handling

### What to check

- No bare `except:` or `except Exception:` that silently swallows errors.
  Always log or re-raise.
- Custom exceptions should inherit from a project-specific base or a
  meaningful built-in exception, not plain `Exception`.
- `finally` blocks should not mask exceptions.
- Resource cleanup should use context managers (`with`) not manual
  `try/finally`.

### Tools the agent must use

| Step | Tool / Command | Why |
|---|---|---|
| Find bare excepts | `grep_search` for `except\s*:` and `except Exception\s*:` | Quick pattern match |
| Lint for broad exceptions | `run_in_terminal`: `python -m ruff check --select E722,BLE001 <file>` | E722 = bare except, BLE001 = blind exception |

---

## 7. Security basics

### What to check

- No hard-coded secrets, passwords, tokens, or API keys.
- No use of `eval()`, `exec()`, or `pickle.loads()` on untrusted input.
- No `subprocess` calls with `shell=True` when avoidable.
- File paths from user input must be validated (no path traversal).

### Tools the agent must use

| Step | Tool / Command | Why |
|---|---|---|
| Scan for secrets | `grep_search` for `password\|secret\|token\|api_key\|AWS_` in changed files | Quick surface scan |
| Scan for dangerous calls | `grep_search` for `eval\(\|exec\(\|pickle\.loads\|shell=True` | Catches common anti-patterns |
| Deeper scan | `run_in_terminal`: `python -m bandit -r <file>` | Security linter; install with `pip install bandit` if missing |

---

## 8. Testing impact

### What to check

- If a public function was added or its signature changed, there should be a
  corresponding test or an explicit "Testing" note in the MR description
  explaining why tests are deferred.
- Modified behaviour should not break existing tests—run them to confirm.

### Tools the agent must use

| Step | Tool / Command | Why |
|---|---|---|
| Run existing tests | `run_in_terminal`: `python -m pytest tests/ -x -q --tb=short` | Fast fail-first run |
| Check test coverage of changed files | `run_in_terminal`: `python -m pytest --cov=pycwb --cov-report=term-missing tests/` | Shows untested lines; install `pytest-cov` if missing |
| Find test files | `file_search` for `**/test_*.py` | Locates relevant test modules |

---

## Summary checklist (quick reference)

```
[ ] Docstrings: NumPy style, all public symbols documented
[ ] Type annotations: modern PEP 585/604 syntax, all public signatures typed
[ ] Code style: PEP 8, 120-char lines, ruff clean
[ ] Unused code: no dead imports, variables, or commented-out blocks
[ ] Comments: accurate, explain "why", no stale or bare TODOs
[ ] Error handling: no bare excepts, use context managers
[ ] Security: no secrets, no eval/exec on untrusted input
[ ] Tests: existing tests pass, new public API has tests
```

---

## Tool installation one-liner

If any of the review tools are missing, install them all at once:

```bash
pip install ruff mypy pydocstyle bandit vulture pytest pytest-cov
```