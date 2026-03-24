"""Stub for test coverage endpoint plugin."""

from .base import EndpointPlugin


class TestCoveragePlugin(EndpointPlugin):
    """Test coverage analysis endpoint (stub)."""

    @property
    def name(self) -> str:
        return "test_coverage"

    @property
    def api_path(self) -> str:
        return "/v1/analysis/test-coverage"

    @property
    def baseline_prompt(self) -> str:
        return "Identify untested code in this repository."

    @property
    def enhanced_prompt(self) -> str:
        return "Using the dependency graph, identify untested code."

    def extract_ground_truth(self, repo, pr_number, language="typescript", scope_prefix=None):
        raise NotImplementedError("TestCoveragePlugin.extract_ground_truth not implemented")
