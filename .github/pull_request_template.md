<!--
Full workflow, style, and the agent PR lifecycle (open -> assess -> close out)
are documented in CONTRIBUTING.md. This template is the minimum every PR
(human or agent-authored) should fill in.
-->

## Summary

<!-- 1-3 sentences: what changed and why. Link an issue if one exists. -->

## Type of change

- [ ] Fix (bug fix, no API change)
- [ ] Feature (new functionality, backward compatible)
- [ ] Breaking change (requires a MAJOR version bump)
- [ ] Docs / CLAUDE.md / CHANGELOG only
- [ ] Chore (CI, tooling, refactor with no behavior change)

## Checklist

- [ ] Tests added/updated (both `inplace=True` and `inplace=False` where applicable)
- [ ] `pytest tests/ -v` passes locally
- [ ] Docstrings follow the Google-style pattern in CLAUDE.md (Args/Returns/Raises/Example)
- [ ] No `print()` — uses `logger` per project convention
- [ ] Ran against the `code-review` skill checklist (or equivalent manual review)
- [ ] `CHANGELOG.md` `[Unreleased]` section updated
- [ ] `README.md` updated if this touches public API or usage examples
- [ ] No version bump in this PR (version bumps are a separate release PR/tag — see CONTRIBUTING.md)

## Testing

<!-- Commands run and their results, e.g.:
pytest tests/test_preprocessing.py -v
218 passed
-->

## Breaking changes / migration notes

<!-- Delete this section if not applicable -->
