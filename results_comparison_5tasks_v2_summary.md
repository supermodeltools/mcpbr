# MCP vs Baseline Comparison Results

**Date:** 2026-01-17  
**Model:** claude-sonnet-4-20250514  
**MCP Server:** @supermodeltools/mcp-server  
**Tasks:** 5 (SWE-bench Lite)

## Summary

| Metric | MCP (w/ Supermodel) | Baseline (bare Claude Code) |
|--------|---------------------|----------------------------|
| **Resolved** | 3/5 (60%) | 3/5 (60%) |
| **Improvement** | +0.0% | - |

## Per-Task Breakdown

| Task | MCP | Baseline | MCP `explore_codebase` calls |
|------|-----|----------|------------------------------|
| astropy-6938 | Resolved | Resolved | 2 |
| astropy-14182 | Failed | Failed | 2 |
| astropy-12907 | Resolved | Resolved | 2 |
| astropy-14365 | Failed | Failed | 2 |
| astropy-14995 | Resolved | Resolved | 1 |

## Observations

- MCP tool was used consistently (9 total calls across 5 tasks)
- Same resolution rate for both approaches
- Same tasks failed for both MCP and baseline
- MCP runs used fewer total tool calls in some cases

## Files

- Full results: `results_comparison_5tasks_v2.json`
- Logs: `.logs_comparison_5tasks_v2/`
