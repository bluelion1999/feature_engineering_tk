# Contributing to Feature Engineering Toolkit

This document is the operating manual for anyone — human or Claude Code agent —
making changes to this repository. It covers branch/commit style, the PR
lifecycle (open → assess → close out), and versioning/release practices.

For code-level conventions (docstrings, `inplace` pattern, exceptions,
logging) see `CLAUDE.md` — that is the source of truth for *how code is
written*. This document is the source of truth for *how change moves through
git and GitHub*.

> **Note:** `.claude/` (agent definitions, skills, `settings.json`) is local
> Claude Code tooling and is intentionally **not** tracked in this repo
> (see `.gitignore`). References below to `.claude/agents/pr-manager.md`,
> `release-manager`, `test-writer`, `doc-generator`, or the `code-review`
> skill describe workflow that exists on contributors' local machines, not
> files you'll find after cloning. The workflow and checklists themselves —
> the actual contract — live in this file and are followable by hand or by
> any agent, with or without those local configs.

---

## 1. Branching

One feature/fix per branch. Branch from `master`, name it by intent:

| Prefix | For | Example |
|---|---|---|
| `feature/` | New functionality, backward compatible | `feature/binning-strategies` |
| `fix/` | Bug fixes | `fix/zscore-division-by-zero` |
| `docs/` | README, CLAUDE.md, CHANGELOG-only changes | `docs/update-vif-examples` |
| `chore/` | CI, tooling, dependency bumps, refactors with no behavior change | `chore/add-ci-workflow` |
| `release/` | Version bump + CHANGELOG for a specific release | `release/v2.5.0` |

Avoid vague branch names (`fly_catcher`, `teacher`) — they don't communicate
scope and make stale-branch cleanup harder to reason about later. If a branch
is exploratory/personal, that's fine, but convert it to a proper
`feature/*`/`fix/*` branch before opening a PR.

**Keep branches short-lived.** A branch that's fully merged into `master`
(`git log master..<branch> --oneline` is empty) should be deleted — locally
and on the remote — right after merge. Don't let merged branches accumulate;
they make it hard to tell active work from history.

---

## 2. Commit messages

Imperative mood, one logical change per commit. Match the existing history:

```
Fix: division by zero in class imbalance calculation (#2)
Add: binning suggestions to DataAnalyzer
Release v2.5.0: binning suggestions and CI workflows
Docs: update CHANGELOG for v2.4.1
```

- Prefix with `Fix:`, `Add:`, `Docs:`, `Refactor:`, `Test:`, `Release vX.Y.Z:`
  when the commit is purely one of those things. Prose subjects (no prefix)
  are fine for commits that touch several concerns.
- Reference an issue number in parentheses when one exists.
- Don't bundle a version bump into a feature commit — release commits are
  their own commit (see §5).

---

## 3. The PR lifecycle for agents (open → assess → close out)

This section is written for a Claude Code agent (or a human following the
same discipline) driving a feature from branch to merge. The
`.claude/agents/pr-manager.md` agent automates this end to end; this is the
checklist it — and anyone working manually — follows.

### 3.1 Before opening

1. Branch from an up-to-date `master`.
2. Implement the change following `CLAUDE.md` patterns (inplace pattern,
   custom exceptions, logging, type hints, docstrings).
3. Write/update tests (`test-writer` agent, or by hand) — both `inplace=True`
   and `inplace=False` paths, edge cases, error cases.
4. Run the full suite locally: `pytest tests/ -v`. Don't open a PR on red
   tests.
5. Self-review against the `code-review` skill (10 categories: type hints,
   docstrings, inplace pattern, exceptions, logging, constants, validation,
   style, tests, version compatibility). Fix HIGH-priority findings before
   opening the PR; note MEDIUM/LOW findings in the PR description if
   deferring them.
6. Add an entry under `## [Unreleased]` in `CHANGELOG.md` (create that
   section at the top of the file if it doesn't exist yet — see §5.3).

### 3.2 Opening the PR

```bash
git push -u origin <branch>
gh pr create --title "<concise, imperative>" --body "$(cat <<'EOF'
## Summary
...
EOF
)"
```

- Title mirrors the primary commit's subject.
- Fill out every section of `.github/pull_request_template.md` — don't leave
  checklist items unchecked without a stated reason.
- Target `master` unless explicitly coordinating a multi-PR release branch.
- Let CI run (`.github/workflows/ci.yml`): test matrix across Python
  3.8–3.12, lint, build check, version-consistency check. A red CI run means
  the PR isn't ready for assessment yet — fix it before asking for review.

### 3.3 Assessing the PR

"Assess" means answering, explicitly, in the PR thread or a review comment:

- **Correctness**: does it do what the summary claims? Any regressions in
  adjacent methods sharing the same base class / utility functions?
- **Scope**: is this one logical change, or should it split? (Mixing a
  version bump with a feature, or a refactor with a bug fix, should be
  flagged and split.)
- **Test coverage**: does the diff match the 3–7-tests-per-feature standard
  in `CLAUDE.md`? Are error paths and edge cases (empty DataFrame, missing
  columns, constant/zero-variance columns) covered?
- **Breaking changes**: does this change a public method's signature,
  default, or return type? If yes, the PR must call it out explicitly and
  the eventual release must be a MAJOR bump (§5.1).
- **Docs**: do README/CLAUDE.md examples still match the code? Is the
  CHANGELOG `[Unreleased]` entry accurate?

Record the verdict as a normal PR review (`gh pr review --approve` /
`--request-changes` / `--comment`). Don't merge your own PR silently — even
an agent-authored PR should get an explicit approve step, and a human should
be the one who merges anything touching public API behavior.

### 3.4 Closing out (after merge)

A PR is not "done" at merge — done means:

1. **Delete the branch**, local and remote:
   ```bash
   git push origin --delete <branch>
   git branch -d <branch>
   ```
   (`gh pr merge --delete-branch` does both in one step.)
2. **Confirm `master` is green** — CI on the merge commit passed, not just
   the PR branch.
3. **Leave `CHANGELOG.md`'s `[Unreleased]` section as the record** of what
   shipped — don't assign it a version number yet; that happens at release
   time (§5).
4. **Close the linked issue**, if any, with a one-line summary of the
   resolution.
5. If the change affects agent-facing conventions (a new pattern, a new
   threshold, a new exception type), **update `CLAUDE.md`** in the same PR
   or a prompt-fast follow-up — stale project memory is worse than no memory.

---

## 4. Stale branch hygiene

Periodically (or when asked to "clean up branches"):

```bash
git fetch --prune
for b in $(git for-each-ref --format='%(refname:short)' refs/heads/); do
  echo "$b: $(git log master.."$b" --oneline | wc -l) commits ahead of master"
done
```

- **0 commits ahead** → fully merged, safe to delete.
- **N commits ahead, no open PR** → either open a PR for it or confirm with
  whoever owns it that it's abandoned before deleting. Never force-delete
  someone's unmerged work without asking.

---

## 5. Versioning & releases

Feature Engineering Toolkit follows [SemVer 2.0.0](https://semver.org/):
`MAJOR.MINOR.PATCH`.

### 5.1 Choosing the bump

- **MAJOR**: breaking change to a public method's signature, default
  behavior, or return type (e.g. the v2.0.0 `inplace` default flip).
- **MINOR**: new method/class/module, backward compatible.
- **PATCH**: bug fix, no API change.

### 5.2 The 5 version locations

These must always agree — CI enforces this (`version-consistency` job in
`ci.yml`, and again as a hard gate in `release.yml` before a tag can produce
a GitHub Release):

1. `pyproject.toml` — `version = "X.Y.Z"`
2. `setup.py` — `version='X.Y.Z'`
3. `feature_engineering_tk/__init__.py` — `__version__ = 'X.Y.Z'`
4. `README.md` — `# Feature Engineering Toolkit vX.Y.Z` (line 1)
5. `CHANGELOG.md` — top entry `## [X.Y.Z] - YYYY-MM-DD`

The `release-manager` agent owns updating all five in one commit. If you
bump manually, grep for the old version string across the repo before
committing:

```bash
grep -rn "2\.4\.3" --include="*.py" --include="*.md" --include="*.toml" .
```

### 5.3 `[Unreleased]` workflow

`CHANGELOG.md` should always have an `## [Unreleased]` section at the top
that PRs append to as they merge (see §3.1/§3.4). At release time, the
`release-manager` agent renames `[Unreleased]` to `[X.Y.Z] - YYYY-MM-DD` and
opens a fresh empty `[Unreleased]` section above it. This keeps the
changelog accurate between releases instead of being reconstructed from git
log after the fact.

### 5.4 Cutting a release

1. Confirm `master` is green and `[Unreleased]` reflects everything merged
   since the last tag.
2. Invoke the `release-manager` agent (or follow its checklist manually) to
   bump all 5 locations and finalize the CHANGELOG section.
3. Commit: `Release vX.Y.Z: <one-line summary>`.
4. **Ask the user before tagging or pushing** — this is a hard rule for the
   `release-manager` agent, not a suggestion.
5. Tag: `git tag -a vX.Y.Z -m "Release vX.Y.Z\n\n<summary>"`.
6. Push commit and tag: `git push && git push --tags`.
7. Pushing the tag triggers `.github/workflows/release.yml`, which:
   re-verifies the 5 locations against the tag, reruns the full test suite,
   builds sdist/wheel, and publishes a GitHub Release with the matching
   CHANGELOG section attached.
8. PyPI publishing is a separate, gated step (`publish-pypi` job, behind a
   GitHub Environment requiring manual approval) — it stays off until you
   explicitly configure a
   [PyPI Trusted Publisher](https://docs.pypi.org/trusted-publishers/) for
   this repo. Until then, PyPI release remains a manual
   `python -m build && twine upload dist/*` step, same as today.

### 5.5 Hotfixes

Branch `fix/*` directly off `master` (not off an in-flight feature branch),
PATCH bump only, minimal diff, fast-tracked through §3 with the same CI
gates — no shortcuts on tests just because it's urgent.

### 5.6 Pre-releases

Use SemVer pre-release suffixes: `2.5.0a1` (alpha), `2.5.0b1` (beta),
`2.5.0rc1` (release candidate). `release.yml` automatically marks a GitHub
Release as a "prerelease" when the tag contains `a`, `b`, or `rc`.

---

## 6. CI reference

| Workflow | Trigger | Does |
|---|---|---|
| `ci.yml` | Every push/PR to `master` | Test matrix (Py 3.8–3.12), flake8 + black, package build check, 5-location version-consistency check |
| `release.yml` | Push of a `vX.Y.Z` tag | Re-verifies version consistency against the tag, full test run, builds dist, publishes GitHub Release, optional gated PyPI publish |

A PR cannot be merged with a red `ci.yml` run. If branch protection isn't
already configured to require it, do so in Settings → Branches → require
status checks `test`, `lint`, `build`, `version-consistency` before merging
to `master`.
