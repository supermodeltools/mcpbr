"""Stub for circular dependency analysis endpoint plugin."""

from .base import EndpointPlugin


class CircularDepsPlugin(EndpointPlugin):
    """Circular dependency analysis endpoint (stub)."""

    @property
    def name(self) -> str:
        return "circular_deps"

    @property
    def api_path(self) -> str:
        return "/v1/analysis/circular-deps"

    @property
    def baseline_prompt(self) -> str:
        return "Find all circular dependencies in this repository."

    @property
    def enhanced_prompt(self) -> str:
        return "Using the dependency graph, identify all circular dependencies."

    def extract_ground_truth(self, repo, pr_number, language="typescript", scope_prefix=None):
        raise NotImplementedError("CircularDepsPlugin.extract_ground_truth not implemented")
