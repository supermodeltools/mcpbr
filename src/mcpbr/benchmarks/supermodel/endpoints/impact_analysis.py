"""Stub for impact analysis endpoint plugin."""

from .base import EndpointPlugin


class ImpactAnalysisPlugin(EndpointPlugin):
    """Impact analysis endpoint (stub)."""

    @property
    def name(self) -> str:
        return "impact_analysis"

    @property
    def api_path(self) -> str:
        return "/v1/analysis/impact"

    @property
    def baseline_prompt(self) -> str:
        return "Analyze the impact of changes in this repository."

    @property
    def enhanced_prompt(self) -> str:
        return "Using the dependency graph, analyze the impact of changes."

    def extract_ground_truth(self, repo, pr_number, language="typescript", scope_prefix=None):
        raise NotImplementedError("ImpactAnalysisPlugin.extract_ground_truth not implemented")
