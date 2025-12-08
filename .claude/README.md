# MLToolkit Claude Code Configuration

This directory contains custom agents, skills, and settings for optimizing Claude Code workflows in the MLToolkit project.

## Directory Structure

```
.claude/
├── agents/              # Custom subagents for specialized tasks
│   ├── test-writer.md       # Write comprehensive pytest tests
│   ├── release-manager.md   # Manage version bumps and releases
│   └── doc-generator.md     # Generate comprehensive docstrings
├── skills/              # Auto-activating capabilities
│   └── code-review/
│       └── SKILL.md         # Automatic code quality review
├── settings.json        # Project settings and permissions
└── README.md           # This file
```

## Custom Agents

### 1. test-writer

**Purpose**: Write comprehensive pytest tests following MLToolkit patterns

**Invoke with**:
- "Write tests for this method"
- "Add test coverage for the new feature"
- "Use the test-writer agent to create tests"

**What it does**:
- Reads existing tests to match style
- Creates tests for both inplace=True and inplace=False
- Covers edge cases (empty DataFrame, missing columns)
- Tests error handling with custom exceptions
- Runs tests immediately after writing
- Ensures 182+ test baseline maintained

**Key features**:
- Understands MLToolkit test patterns from CLAUDE.md
- Groups tests in classes by functionality
- Uses descriptive test names: `test_<method>_<scenario>`
- Validates with pytest after writing

### 2. release-manager

**Purpose**: Manage version bumps, CHANGELOG updates, and release preparation

**Invoke with**:
- "Prepare release v2.2.1"
- "Use the release-manager agent to bump version"
- "Create a patch release for the bug fix"

**What it does**:
- Follows semantic versioning strictly
- Updates version in all 5 required files
- Maintains CHANGELOG.md in Keep a Changelog format
- Runs full test suite before release
- Creates git tags with annotations
- **Never pushes without user confirmation**

**Key features**:
- Ensures version consistency across setup.py, __init__.py, README.md, CHANGELOG.md
- Distinguishes between MAJOR (breaking), MINOR (features), PATCH (fixes)
- Quality gates: tests passing, version consistency, CHANGELOG completeness
- Handles hotfixes and pre-releases (alpha/beta/rc)

### 3. doc-generator

**Purpose**: Generate comprehensive docstrings following Google Style

**Invoke with**:
- "Document this method"
- "Add docstrings to these functions"
- "Use the doc-generator agent to improve documentation"

**What it does**:
- Writes Google-style docstrings with Args/Returns/Raises/Example sections
- Documents inplace parameter behavior clearly
- Uses MLToolkit custom exceptions
- Adds realistic, runnable examples
- Ensures type hints in signature (not docstring)

**Key features**:
- Understands v2.0.0+ inplace pattern documentation requirements
- Knows which methods need Example sections (API Reference methods)
- Uses imperative mood for brief descriptions
- Matches existing docstring style in codebase

## Auto-Activating Skill

### code-review

**Purpose**: Automatically review code for quality standards

**Auto-activates when**: Making code changes, editing files, reviewing PRs

**What it checks**:
1. **Type Hints**: Complete type annotations
2. **Documentation**: Comprehensive docstrings
3. **Inplace Pattern**: v2.2.0+ compliance (returns self)
4. **Exception Handling**: Custom exceptions used
5. **Logging**: logger instead of print()
6. **Constants**: Magic numbers extracted
7. **Validation**: Proper input checking
8. **Code Style**: Pythonic patterns, DRY principle
9. **Testing**: Coverage expectations (3-7 tests per feature)
10. **Version Compatibility**: v2.0.0+ and v2.2.0+ standards

**Output**: Prioritized feedback (HIGH/MEDIUM/LOW) with specific fixes

## Project Settings

`settings.json` configures:
- **Model**: Sonnet (default for MLToolkit)
- **Permissions**:
  - ✅ Allowed: Read, Write, Edit, Bash, Grep, Glob
  - ❌ Denied: rm -rf, git push --force, .env access
- **Subagents**: Enabled
- **Environment**: PYTHONPATH, PYTEST_ARGS

## Usage Examples

### Write Tests for a New Method

```bash
# Automatic invocation (Claude recognizes the task)
"I added a new method clean_string_columns() to DataPreprocessor. Write comprehensive tests for it."

# Explicit invocation
"Use the test-writer agent to create tests for the validate_data_quality() method."
```

### Prepare a Release

```bash
# Minor release (new features)
"Prepare release v2.3.0 with the new features we added."

# Patch release (bug fixes)
"Use the release-manager agent to create a patch release v2.2.1 for the bug fixes."
```

### Generate Documentation

```bash
# Single method
"Document the handle_whitespace_variants() method with examples."

# Multiple methods
"Use the doc-generator agent to add comprehensive docstrings to all methods in the StringPreprocessing class."
```

### Code Review (Auto-Activates)

```bash
# The code-review skill automatically activates when you're editing code
# No explicit invocation needed - it watches for code changes

# Or request explicitly:
"Review this code against MLToolkit standards."
```

## Integration with CLAUDE.md

These agents and skills work in tandem with the project documentation in `CLAUDE.md`:

- **CLAUDE.md**: Project context, patterns, conventions (what Claude reads)
- **.claude/ agents**: Specialized AI assistants (how Claude works)
- **.claude/ skills**: Auto-activating capabilities (proactive assistance)
- **.claude/ settings.json**: Project configuration (permissions, model)

## Customization

### Adding New Agents

Create `.claude/agents/my-agent.md`:

```yaml
---
name: my-agent
description: When to invoke this agent (natural language)
tools: Read, Write, Bash
model: sonnet
---

Your agent's system prompt here...
Instructions, patterns, examples...
```

### Adding New Skills

Create `.claude/skills/my-skill/SKILL.md`:

```yaml
---
name: my-skill
description: Auto-activates when... (natural language trigger)
allowed-tools: Read, Grep
---

Skill instructions...
What to watch for, how to help...
```

### Modifying Settings

Edit `.claude/settings.json`:
- Add/remove tool permissions
- Change default model
- Set environment variables
- Configure subagent behavior

## Best Practices

1. **Let agents specialize**: Don't try to do everything in one agent
2. **Use natural invocation**: "Write tests for X" works better than "/test-writer X"
3. **Skills are proactive**: They activate automatically based on context
4. **Check CLAUDE.md first**: It's the source of truth for project patterns
5. **Agents inherit from CLAUDE.md**: They see the project context automatically

## Troubleshooting

**Agent not invoking?**
- Check the `description` field - it should match the task you're requesting
- Try explicit invocation: "Use the [agent-name] agent to..."

**Skill not activating?**
- Skills activate based on relevance detection
- Make the `description` field more specific
- Consider using an agent instead for explicit control

**Permission denied?**
- Check `.claude/settings.json` permissions
- Add the required tool to the `allow` list

## Resources

- **Claude Code Docs**: https://code.claude.com/docs
- **Subagents**: https://code.claude.com/docs/en/sub-agents
- **Skills**: https://code.claude.com/docs/en/skills
- **Settings**: https://code.claude.com/docs/en/settings

---

**Created**: 2025-12-07
**MLToolkit Version**: v2.2.0
**Agents**: 3 (test-writer, release-manager, doc-generator)
**Skills**: 1 (code-review)
