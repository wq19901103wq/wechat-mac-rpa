# Contributing to wechat-mac-rpa

First off, thanks for taking the time to contribute! 🎉

This document outlines how to contribute to the project.

## Ways to Contribute

- 🐛 **Report bugs** — Open an issue with a clear description and steps to reproduce
- 💡 **Suggest features** — Share your ideas in Discussions or Issues
- 📝 **Improve documentation** — Fix typos, add examples, translate docs
- 🔧 **Submit code** — Fix bugs, add features, improve performance
- 🎯 **Add Skills** — Write new Markdown skill cards for new use cases
- ⭐ **Star the repo** — Help others discover the project

## Getting Started

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/<your-username>/wechat-mac-rpa.git`
3. **Create a branch**: `git checkout -b feature/your-feature-name`
4. **Make your changes**
5. **Run tests**: `python3 -m pytest src/tests -v`
6. **Commit**: Use clear, descriptive commit messages
7. **Push** and open a Pull Request

## Development Setup

```bash
# Clone the repo
git clone https://github.com/example-owner/wechat-mac-rpa.git
cd wechat-mac-rpa

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements-dev.txt

# Copy config template
cp .env.example .env
# Edit .env with your API keys

# Run tests
python3 -m pytest src/tests/ -v
```

## Code Style

- Follow existing code patterns in the codebase
- Add type annotations for all public functions
- Keep functions focused — single responsibility
- Add docstrings for complex logic

## Writing Skills

Skills are Markdown files in `skills/`. Each skill card should include:

```markdown
---
name: your_skill_name
triggers:
  - keyword1
  - keyword2
---

# Skill Title

## When to use this skill
Describe the trigger conditions.

## Control flow
Step-by-step instructions for the bot.

## Output rules
How the bot should format its response.
```

## Pull Request Guidelines

- Keep PRs focused — one feature or fix per PR
- Include a clear description of what changed and why
- If adding a new feature, update relevant documentation
- Make sure all tests pass before submitting
- If your PR addresses an issue, reference it: `Fixes #123`

## Reporting Bugs

When reporting a bug, please include:

- macOS version and WeChat version
- Python version
- Steps to reproduce
- Expected vs actual behavior
- Relevant log output (remove sensitive info)

## Questions?

Feel free to open a Discussion or an Issue. We're happy to help!

---

By participating in this project, you agree to abide by our code of conduct: be respectful, constructive, and helpful.
