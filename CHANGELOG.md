# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-17

### Added

- Initial release of mcpbr (Model Context Protocol Benchmark Runner)
- Support for multiple LLM providers:
  - OpenRouter (multi-provider gateway)
  - OpenAI (direct API)
  - Anthropic (direct API)
  - Google AI / Gemini (direct API)
- Agent harness implementations:
  - Claude Code CLI with MCP server support
  - OpenAI Codex CLI
  - OpenCode CLI
  - Gemini CLI
- Docker-based task isolation for SWE-bench evaluation
- Baseline agent for comparison (no tools)
- JSON and Markdown report generation
- Configurable evaluation parameters via YAML
- CLI commands: `run`, `init`, `models`, `harnesses`, `providers`
- Real-time streaming output when using `--verbose` flag with Claude Code harness
- Tool usage tracking: counts tool calls by name and includes breakdown in results JSON
- Each streamed output line is prefixed with task instance ID for parallel worker disambiguation
- **In-container agent execution**: The Claude Code CLI now runs inside the Docker container where all dependencies are installed. This ensures Python imports work correctly (e.g., `from astropy import ...`) and the agent can verify fixes by running tests.
- Pre-built SWE-bench Docker images from Epoch AI's registry (`ghcr.io/epoch-research/swe-bench.eval`) are now used when available, providing:
  - Repository at the correct commit with all dependencies pre-installed
  - Consistent, reproducible evaluation environments
  - x86_64 images with automatic emulation on ARM64 (Apple Silicon) Macs
- Timestamped log filenames to prevent overwrites: `{instance_id}_{run_type}_{YYYYMMDD_HHMMSS}.json`
- `--no-prebuilt` CLI flag to disable pre-built images and build from scratch
- Network access for containers to enable API calls from within Docker

[0.1.0]: https://github.com/greynewell/mcpbr/releases/tag/v0.1.0
