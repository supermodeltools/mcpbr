"""Tests for benchmark preflight validation system."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcpbr.benchmark_preflight import (
    PreflightReport,
    PreflightResult,
    _check_single_instance,
    run_benchmark_preflight,
)


class TestPreflightResult:
    """Tests for PreflightResult data structure."""

    def test_basic_construction(self) -> None:
        result = PreflightResult(
            instance_id="test-123",
            status="passed",
            fail_to_pass_passed=3,
            fail_to_pass_total=3,
            pass_to_pass_passed=5,
            pass_to_pass_total=5,
            language="python",
        )
        assert result.instance_id == "test-123"
        assert result.status == "passed"
        assert result.error is None

    def test_failed_result(self) -> None:
        result = PreflightResult(
            instance_id="test-456",
            status="failed",
            fail_to_pass_passed=1,
            fail_to_pass_total=3,
            error="fail_to_pass: 1/3 passed",
            language="go",
        )
        assert result.status == "failed"
        assert result.error is not None

    def test_error_result(self) -> None:
        result = PreflightResult(
            instance_id="test-789",
            status="error",
            error="Docker connection failed",
        )
        assert result.status == "error"
        assert result.language == "unknown"


class TestPreflightReport:
    """Tests for PreflightReport aggregate results."""

    def test_all_passed(self) -> None:
        report = PreflightReport(total=3, passed=3, failed=0, errors=0)
        assert report.success_rate == 100.0

    def test_empty_report(self) -> None:
        report = PreflightReport(total=0, passed=0, failed=0, errors=0)
        assert report.success_rate == 0.0

    def test_partial_success(self) -> None:
        report = PreflightReport(total=10, passed=7, failed=2, errors=1)
        assert report.success_rate == 70.0

    def test_all_failed(self) -> None:
        report = PreflightReport(total=5, passed=0, failed=5, errors=0)
        assert report.success_rate == 0.0

    def test_default_results_list(self) -> None:
        report = PreflightReport()
        assert report.results == []


class TestCheckSingleInstance:
    """Tests for single instance preflight check."""

    @pytest.mark.asyncio
    async def test_successful_check(self) -> None:
        mock_env = MagicMock()
        mock_env.uses_prebuilt = True
        mock_env.cleanup = AsyncMock()
        mock_env.exec_command = AsyncMock(return_value=(0, "", ""))

        mock_benchmark = MagicMock()
        mock_benchmark.create_environment = AsyncMock(return_value=mock_env)

        task = {
            "instance_id": "django__django-16046",
            "repo": "django/django",
            "repo_language": "python",
            "patch": "diff --git a/fix.py",
            "test_patch": "",
            "fail_to_pass": '["test_one"]',
            "pass_to_pass": '["test_two"]',
        }

        mock_docker = MagicMock()

        with (
            patch("mcpbr.benchmark_preflight.apply_patch", new_callable=AsyncMock) as mock_apply,
            patch("mcpbr.benchmark_preflight.run_tests", new_callable=AsyncMock) as mock_tests,
        ):
            mock_apply.return_value = (True, "")
            # fail_to_pass: 1/1 passed, pass_to_pass: 1/1 passed
            mock_tests.side_effect = [
                MagicMock(passed=1, total=1),
                MagicMock(passed=1, total=1),
            ]

            result = await _check_single_instance(mock_benchmark, task, mock_docker)

        assert result.status == "passed"
        assert result.instance_id == "django__django-16046"
        assert result.language == "python"
        mock_env.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_patch_apply_failure(self) -> None:
        mock_env = MagicMock()
        mock_env.uses_prebuilt = True
        mock_env.cleanup = AsyncMock()

        mock_benchmark = MagicMock()
        mock_benchmark.create_environment = AsyncMock(return_value=mock_env)

        task = {
            "instance_id": "test-fail",
            "repo": "org/repo",
            "repo_language": "go",
            "patch": "bad patch",
            "fail_to_pass": '["test"]',
            "pass_to_pass": "[]",
        }

        mock_docker = MagicMock()

        with patch("mcpbr.benchmark_preflight.apply_patch", new_callable=AsyncMock) as mock_apply:
            mock_apply.return_value = (False, "Patch does not apply")

            result = await _check_single_instance(mock_benchmark, task, mock_docker)

        assert result.status == "failed"
        assert "Golden patch failed to apply" in (result.error or "")

    @pytest.mark.asyncio
    async def test_no_golden_patch(self) -> None:
        mock_env = MagicMock()
        mock_env.uses_prebuilt = True
        mock_env.cleanup = AsyncMock()

        mock_benchmark = MagicMock()
        mock_benchmark.create_environment = AsyncMock(return_value=mock_env)

        task = {
            "instance_id": "no-patch",
            "repo": "org/repo",
            "repo_language": "python",
            "patch": "",
            "fail_to_pass": '["test"]',
            "pass_to_pass": "[]",
        }

        mock_docker = MagicMock()
        result = await _check_single_instance(mock_benchmark, task, mock_docker)

        assert result.status == "error"
        assert "No golden patch" in (result.error or "")

    @pytest.mark.asyncio
    async def test_exception_handling(self) -> None:
        mock_benchmark = MagicMock()
        mock_benchmark.create_environment = AsyncMock(
            side_effect=RuntimeError("Docker not available")
        )

        task = {
            "instance_id": "error-task",
            "repo": "org/repo",
            "patch": "diff",
        }

        mock_docker = MagicMock()
        result = await _check_single_instance(mock_benchmark, task, mock_docker)

        assert result.status == "error"
        assert "Docker not available" in (result.error or "")


class TestRunBenchmarkPreflight:
    """Tests for the main preflight runner."""

    @pytest.mark.asyncio
    async def test_concurrent_execution(self) -> None:
        mock_benchmark = MagicMock()
        mock_env = MagicMock()
        mock_env.uses_prebuilt = True
        mock_env.cleanup = AsyncMock()
        mock_env.exec_command = AsyncMock(return_value=(0, "", ""))
        mock_benchmark.create_environment = AsyncMock(return_value=mock_env)

        tasks = [
            {
                "instance_id": f"task-{i}",
                "repo": "org/repo",
                "repo_language": "python",
                "patch": "diff --git",
                "test_patch": "",
                "fail_to_pass": '["test"]',
                "pass_to_pass": "[]",
            }
            for i in range(3)
        ]

        mock_docker = MagicMock()

        with (
            patch("mcpbr.benchmark_preflight.apply_patch", new_callable=AsyncMock) as mock_apply,
            patch("mcpbr.benchmark_preflight.run_tests", new_callable=AsyncMock) as mock_tests,
        ):
            mock_apply.return_value = (True, "")
            mock_tests.return_value = MagicMock(passed=1, total=1)

            report = await run_benchmark_preflight(
                benchmark=mock_benchmark,
                tasks=tasks,
                docker_manager=mock_docker,
                max_concurrent=2,
            )

        assert report.total == 3
        assert report.passed == 3
        assert report.failed == 0
        assert report.success_rate == 100.0

    @pytest.mark.asyncio
    async def test_fail_fast(self) -> None:
        call_count = 0

        async def mock_check(
            benchmark: object, task: dict, docker: object, timeout: int = 300
        ) -> PreflightResult:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return PreflightResult(
                    instance_id=task["instance_id"],
                    status="failed",
                    error="Test failure",
                    language="python",
                )
            return PreflightResult(
                instance_id=task["instance_id"],
                status="passed",
                language="python",
            )

        tasks = [{"instance_id": f"task-{i}", "repo": "r", "patch": "d"} for i in range(5)]

        mock_docker = MagicMock()
        mock_benchmark = MagicMock()

        with patch(
            "mcpbr.benchmark_preflight._check_single_instance",
            side_effect=mock_check,
        ):
            report = await run_benchmark_preflight(
                benchmark=mock_benchmark,
                tasks=tasks,
                docker_manager=mock_docker,
                fail_fast=True,
            )

        # Should stop after the failure (task 2)
        assert report.total == 5
        assert report.passed == 1
        assert report.failed == 1
        assert len(report.results) == 2

    @pytest.mark.asyncio
    async def test_error_handling_in_gather(self) -> None:
        mock_benchmark = MagicMock()

        async def failing_create(*args: object, **kwargs: object) -> None:
            raise RuntimeError("Docker error")

        mock_benchmark.create_environment = AsyncMock(side_effect=failing_create)

        tasks = [
            {
                "instance_id": "err-task",
                "repo": "org/repo",
                "patch": "diff",
                "fail_to_pass": '["test"]',
                "pass_to_pass": "[]",
            }
        ]

        mock_docker = MagicMock()

        report = await run_benchmark_preflight(
            benchmark=mock_benchmark,
            tasks=tasks,
            docker_manager=mock_docker,
        )

        assert report.total == 1
        assert report.errors == 1
        assert report.success_rate == 0.0


class TestGetTestListField:
    """Tests for the get_test_list_field helper."""

    def test_lowercase_field(self) -> None:
        from mcpbr.evaluation import get_test_list_field

        task = {"fail_to_pass": '["test_a"]'}
        assert get_test_list_field(task, "fail_to_pass") == '["test_a"]'

    def test_uppercase_field(self) -> None:
        from mcpbr.evaluation import get_test_list_field

        task = {"FAIL_TO_PASS": '["test_b"]'}
        assert get_test_list_field(task, "fail_to_pass") == '["test_b"]'

    def test_lowercase_preferred(self) -> None:
        from mcpbr.evaluation import get_test_list_field

        task = {"fail_to_pass": '["lower"]', "FAIL_TO_PASS": '["upper"]'}
        assert get_test_list_field(task, "fail_to_pass") == '["lower"]'

    def test_missing_field(self) -> None:
        from mcpbr.evaluation import get_test_list_field

        task = {"something_else": "value"}
        assert get_test_list_field(task, "fail_to_pass") == "[]"
