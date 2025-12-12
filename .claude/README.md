# Feature Engineering Toolkit Claude Code Configuration

This directory contains custom agents, skills, and settings for optimizing Claude Code workflows in the Feature Engineering Toolkit project.

## Directory Structure

```
.claude/
├── agents/              # Custom subagents for specialized tasks
│   ├── 🧪 test-writer.md       # Write comprehensive pytest tests
│   ├── 📦 release-manager.md   # Manage version bumps and releases
│   ├── 📝 doc-generator.md     # Generate comprehensive docstrings
│   └── 👨‍🏫 teacher.md           # Onboard users and guide contributors
├── skills/              # Auto-activating capabilities
│   └── 🔍 code-review/
│       └── SKILL.md         # Automatic code quality review
├── settings.json        # Project settings and permissions
└── README.md           # This file
```

## Quick Reference Guide

| Agent/Skill | Icon | Purpose | When to Use |
|-------------|------|---------|-------------|
| **test-writer** | 🧪 | Write pytest tests | Adding features, fixing bugs |
| **release-manager** | 📦 | Manage releases | Version bumps, releases |
| **doc-generator** | 📝 | Generate docs & homogenize | Documenting, version audits |
| **teacher** | 👨‍🏫 | Teach usage & guide contributors | Onboarding, learning, contributing |
| **code-review** | 🔍 | Quality review | Auto-activates on edits |

## Color Coding System

Each agent and skill has a unique visual identifier for quick recognition:

- 🧪 **Purple/Testing** - test-writer: Testing and quality assurance
- 📦 **Blue/Package** - release-manager: Versioning and deployment
- 📝 **Green/Documentation** - doc-generator: API documentation
- 👨‍🏫 **Orange/Teaching** - teacher: User onboarding and contributor guidance
- 🔍 **Yellow/Review** - code-review: Proactive code quality

These icons appear throughout the documentation for easy visual scanning.

## Custom Agents

### 🧪 test-writer (Testing & Quality Assurance)

**Purpose**: Write comprehensive pytest tests following Feature Engineering Toolkit patterns

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
- Understands Feature Engineering Toolkit test patterns from CLAUDE.md
- Groups tests in classes by functionality
- Uses descriptive test names: `test_<method>_<scenario>`
- Validates with pytest after writing

### 📦 release-manager (Release & Version Control)

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

### 📝 doc-generator (Documentation & API Reference)

**Purpose**: Generate comprehensive docstrings and homogenize ALL project documentation

**Invoke with**:
- "Document this method"
- "Homogenize all documentation"
- "Check for version consistency across docs"
- "Use the doc-generator agent to improve documentation"

**What it does**:
- Writes Google-style docstrings with Args/Returns/Raises/Example sections
- Documents inplace parameter behavior clearly
- Uses Feature Engineering Toolkit custom exceptions
- Adds realistic, runnable examples
- Ensures type hints in signature (not docstring)
- **NEW**: Ensures version consistency across ALL files (v2.3.0)
- **NEW**: Validates project name consistency ("Feature Engineering Toolkit")
- **NEW**: Homogenizes formatting across README, CLAUDE.md, CHANGELOG
- **NEW**: Cross-validates documentation references

**Key features**:
- Understands v2.0.0+ inplace pattern documentation requirements
- Knows which methods need Example sections (API Reference methods)
- Uses imperative mood for brief descriptions
- Matches existing docstring style in codebase
- **Proactive homogenization**: Watches for version/naming inconsistencies
- **5-file version validation**: setup.py, __init__.py, README.md, CHANGELOG.md, pyproject.toml
- **No MLToolkit references**: Enforces "Feature Engineering Toolkit" naming

### 👨‍🏫 teacher (Learning & Contribution Guide)

**Purpose**: Teach users how to use the package and guide developers on how to contribute

**Invoke with**:
- "How do I use this package?"
- "Explain how DataPreprocessor works"
- "What should I contribute to?"
- "Show me how to add a new feature"
- "I'm new here, where should I start?"

**What it does**:
- **For Users**: Provides tutorials, examples, and learning paths
- **For Developers**: Explains architecture and identifies contribution opportunities
- Teaches the 5 core classes (DataPreprocessor, FeatureEngineer, DataAnalyzer, TargetAnalyzer, FeatureSelector)
- Explains key concepts (inplace pattern, method chaining, workflows)
- Shows common use cases with complete code examples
- Identifies good first issues based on skill level and interests
- Guides through contribution workflow (setup → implement → test → PR)

**Key features**:
- **Progressive learning**: Beginner → Intermediate → Advanced paths
- **Contribution matching**: Suggests work based on interests and skill level
- **Complete examples**: All code is runnable and practical
- **Skill-based guidance**: Different paths for documentation, testing, features, performance
- **Opportunity identification**: Points to specific files and methods needing work
- **Onboarding focus**: Makes learning and contributing feel accessible

**Contribution categories**:
- 🟢 **Beginner**: Documentation, tests, error messages, simple methods
- 🟡 **Intermediate**: Performance optimization, new features, visualizations
- 🔴 **Advanced**: Pipeline integration, streaming data, AutoML, GPU acceleration

## Auto-Activating Skill

### 🔍 code-review (Proactive Quality Guardian)

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
- **Model**: Sonnet (default for Feature Engineering Toolkit)
- **Permissions**:
  - ✅ Allowed: Read, Write, Edit, Bash, Grep, Glob
  - ❌ Denied: rm -rf, git push --force, .env access
- **Subagents**: Enabled
- **Environment**: PYTHONPATH, PYTEST_ARGS

## Usage Examples

### 🧪 Write Tests for a New Method

```bash
# Automatic invocation (Claude recognizes the task)
"I added a new method clean_string_columns() to DataPreprocessor. Write comprehensive tests for it."

# Explicit invocation
"Use the test-writer agent to create tests for the validate_data_quality() method."
```

### 📦 Prepare a Release

```bash
# Minor release (new features)
"Prepare release v2.3.0 with the new features we added."

# Patch release (bug fixes)
"Use the release-manager agent to create a patch release v2.2.1 for the bug fixes."
```

### 📝 Generate Documentation

```bash
# Single method
"Document the handle_whitespace_variants() method with examples."

# Multiple methods
"Use the doc-generator agent to add comprehensive docstrings to all methods in the StringPreprocessing class."

# Homogenize all documentation (NEW)
"Homogenize all documentation - check version consistency and project naming."

# Version audit (NEW)
"Check if all files use v2.3.0 consistently."

# Project name validation (NEW)
"Make sure no files reference MLToolkit instead of Feature Engineering Toolkit."
```

### 👨‍🏫 Learn & Contribute

```bash
# For new users - learning how to use the package
"How do I use this package?"
"Explain how DataPreprocessor works with examples"
"Show me how to prepare data for machine learning"

# For beginners - understanding concepts
"What's the inplace pattern and when should I use it?"
"Explain method chaining in this library"

# For contributors - finding where to help
"What should I contribute to?"
"I'm a Python beginner, where can I help?"
"Show me good first issues for intermediate developers"

# For contributors - learning how to contribute
"How do I add a new feature?"
"Explain the codebase architecture"
"What are the contribution guidelines?"
```

### 🔍 Code Review (Auto-Activates)

```bash
# The code-review skill automatically activates when you're editing code
# No explicit invocation needed - it watches for code changes

# Or request explicitly:
"Review this code against Feature Engineering Toolkit standards."
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
**Last Updated**: 2025-12-11
**Feature Engineering Toolkit Version**: v2.3.0
**Agents**: 4 (🧪 test-writer, 📦 release-manager, 📝 doc-generator, 👨‍🏫 teacher)
**Skills**: 1 (🔍 code-review)
