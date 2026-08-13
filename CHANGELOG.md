# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Multi-language support (in progress)
- Windows support exploration (in progress)

## [0.1.0] - 2024-01-01

### Added
- Core visual RPA engine with multimodal AI vision perception
- Memory system with vector-indexed chat history (BGE embeddings)
- Digital Twin feature for mimicking user's chatting style
- Smart Home Skill (Tuya device integration)
- 3D Printer Management Skill
- Multi-agent support (OpenClaw, Hermes, and other agent frameworks)
- WeChat auto-reply with LLM-powered response generation
- Screenshot-based UI element detection
- AppleScript and Accessibility API for UI interaction
- Configuration via YAML
- Homebrew installation guide
- English README (README_EN.md)
- Contributing guidelines (CONTRIBUTING.md)
- Issue templates (Bug Report, Feature Request, Skill Request)
- Code of Conduct
- Security Policy
- GitHub Pages landing page

### Technical Details
- **No protocol reverse engineering**: Uses visual perception instead
- **No database access**: Does not read WeChat's SQLite databases
- **macOS native**: Built with AppleScript and Accessibility API
- **Extensible Skill system**: Plugin architecture for custom automations
