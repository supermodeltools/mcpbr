"""Tests for SWE-bench Pro benchmark implementation."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcpbr.benchmarks.swebench_pro import (
    PRO_LANGUAGES,
    SWEBENCH_PRO_IMAGE_PREFIX,
    SWEBenchProBenchmark,
    _get_instance_scripts,
    _match_test_results,
    _parse_test_output_locally,
)
from mcpbr.evaluation import TestResults


class TestSWEBenchProInit:
    """Tests for SWEBenchProBenchmark initialization."""

    def test_default_dataset(self) -> None:
        benchmark = SWEBenchProBenchmark()
        assert benchmark.dataset == "ScaleAI/SWE-bench_Pro"

    def test_custom_dataset(self) -> None:
        benchmark = SWEBenchProBenchmark(dataset="custom/dataset")
        assert benchmark.dataset == "custom/dataset"

    def test_name(self) -> None:
        benchmark = SWEBenchProBenchmark()
        assert benchmark.name == "swe-bench-pro"

    def test_custom_scripts_cache_dir(self) -> None:
        benchmark = SWEBenchProBenchmark(scripts_cache_dir=Path("/tmp/test-scripts"))
        assert benchmark._scripts_cache_dir == Path("/tmp/test-scripts")


class TestSWEBenchProNormalizeTask:
    """Tests for task normalization."""

    def test_normalize_basic_task(self) -> None:
        task = {
            "instance_id": "django__django-16046",
            "problem_statement": "Fix the bug",
            "repo": "django/django",
            "base_commit": "abc123",
            "fail_to_pass": '["test_one"]',
            "pass_to_pass": '["test_two"]',
            "test_patch": "diff --git a/test.py",
            "repo_language": "python",
        }
        benchmark = SWEBenchProBenchmark()
        bt = benchmark.normalize_task(task)
        assert bt.task_id == "django__django-16046"
        assert bt.problem_statement == "Fix the bug"
        assert bt.repo == "django/django"
        assert bt.commit == "abc123"
        assert bt.metadata["repo_language"] == "python"

    def test_normalize_with_uppercase_fields(self) -> None:
        """Test that uppercase FAIL_TO_PASS/PASS_TO_PASS are handled."""
        task = {
            "instance_id": "test-123",
            "problem_statement": "desc",
            "repo": "org/repo",
            "base_commit": "def456",
            "FAIL_TO_PASS": '["test_a"]',
            "PASS_TO_PASS": '["test_b"]',
        }
        benchmark = SWEBenchProBenchmark()
        bt = benchmark.normalize_task(task)
        assert bt.task_id == "test-123"
        assert bt.metadata["fail_to_pass"] == '["test_a"]'
        assert bt.metadata["pass_to_pass"] == '["test_b"]'

    def test_normalize_missing_language(self) -> None:
        task = {
            "instance_id": "test-456",
            "problem_statement": "desc",
            "repo": "org/repo",
            "base_commit": "ghi789",
        }
        benchmark = SWEBenchProBenchmark()
        bt = benchmark.normalize_task(task)
        assert bt.metadata["repo_language"] == "unknown"

    def test_normalize_go_task(self) -> None:
        task = {
            "instance_id": "gin-gonic__gin-3890",
            "problem_statement": "Fix routing",
            "repo": "gin-gonic/gin",
            "base_commit": "jkl012",
            "fail_to_pass": '["TestRoute"]',
            "pass_to_pass": "[]",
            "repo_language": "go",
        }
        benchmark = SWEBenchProBenchmark()
        bt = benchmark.normalize_task(task)
        assert bt.metadata["repo_language"] == "go"


class TestGetInstanceScripts:
    """Tests for _get_instance_scripts."""

    def test_reads_scripts_from_directory(self, tmp_path: Path) -> None:
        instance_id = "instance_test__repo-abc123"
        instance_dir = tmp_path / "run_scripts" / instance_id
        instance_dir.mkdir(parents=True)

        run_script_content = "#!/bin/bash\necho 'test'"
        parser_content = "import sys\nprint('parser')"

        (instance_dir / "run_script.sh").write_text(run_script_content)
        (instance_dir / "parser.py").write_text(parser_content)

        run_script, parser = _get_instance_scripts(tmp_path, instance_id)
        assert run_script == run_script_content
        assert parser == parser_content

    def test_raises_on_missing_run_script(self, tmp_path: Path) -> None:
        instance_id = "instance_missing__repo-abc123"
        instance_dir = tmp_path / "run_scripts" / instance_id
        instance_dir.mkdir(parents=True)
        (instance_dir / "parser.py").write_text("parser")

        with pytest.raises(FileNotFoundError, match=r"run_script\.sh"):
            _get_instance_scripts(tmp_path, instance_id)

    def test_raises_on_missing_parser(self, tmp_path: Path) -> None:
        instance_id = "instance_missing__repo-abc123"
        instance_dir = tmp_path / "run_scripts" / instance_id
        instance_dir.mkdir(parents=True)
        (instance_dir / "run_script.sh").write_text("script")

        with pytest.raises(FileNotFoundError, match=r"parser\.py"):
            _get_instance_scripts(tmp_path, instance_id)

    def test_raises_on_missing_directory(self, tmp_path: Path) -> None:
        (tmp_path / "run_scripts").mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            _get_instance_scripts(tmp_path, "nonexistent_instance")


class TestMatchTestResults:
    """Tests for _match_test_results."""

    def test_all_tests_pass(self) -> None:
        parsed = TestResults(
            passed=2,
            total=2,
            details=[
                {"test": "TestFoo", "passed": True, "status": "PASSED"},
                {"test": "TestBar", "passed": True, "status": "PASSED"},
            ],
        )
        ftp, ptp = _match_test_results(parsed, ["TestFoo"], ["TestBar"])
        assert ftp.passed == 1
        assert ftp.total == 1
        assert ptp.passed == 1
        assert ptp.total == 1

    def test_fail_to_pass_fails(self) -> None:
        parsed = TestResults(
            passed=1,
            total=2,
            details=[
                {"test": "TestFoo", "passed": False, "status": "FAILED"},
                {"test": "TestBar", "passed": True, "status": "PASSED"},
            ],
        )
        ftp, ptp = _match_test_results(parsed, ["TestFoo"], ["TestBar"])
        assert ftp.passed == 0
        assert ftp.total == 1
        assert ptp.passed == 1
        assert ptp.total == 1

    def test_substring_matching(self) -> None:
        """Tests that weren't found by exact match fall back to substring."""
        parsed = TestResults(
            passed=1,
            total=1,
            details=[
                {
                    "test": "test/database.js | Test database key methods",
                    "passed": True,
                    "status": "PASSED",
                },
            ],
        )
        ftp, _ptp = _match_test_results(
            parsed,
            ["test/database.js | Test database key methods"],
            [],
        )
        assert ftp.passed == 1
        assert ftp.total == 1

    def test_empty_lists(self) -> None:
        parsed = TestResults(passed=0, total=0, details=[])
        ftp, ptp = _match_test_results(parsed, [], [])
        assert ftp.total == 0
        assert ptp.total == 0

    def test_test_not_found(self) -> None:
        parsed = TestResults(
            passed=1,
            total=1,
            details=[
                {"test": "TestFoo", "passed": True, "status": "PASSED"},
            ],
        )
        ftp, _ptp = _match_test_results(parsed, ["TestMissing"], [])
        assert ftp.passed == 0
        assert ftp.total == 1
        assert ftp.details[0]["status"] == "NOT_FOUND"

    def test_multiple_fail_to_pass(self) -> None:
        parsed = TestResults(
            passed=2,
            total=3,
            details=[
                {"test": "TestA", "passed": True, "status": "PASSED"},
                {"test": "TestB", "passed": True, "status": "PASSED"},
                {"test": "TestC", "passed": False, "status": "FAILED"},
            ],
        )
        ftp, _ptp = _match_test_results(parsed, ["TestA", "TestB", "TestC"], [])
        assert ftp.passed == 2
        assert ftp.total == 3


class TestParseTestOutputLocally:
    """Tests for _parse_test_output_locally."""

    def test_parses_mocha_json(self) -> None:
        """Test parsing mocha JSON output (NodeBB style)."""
        mocha_output = json.dumps(
            {
                "passes": [
                    {"file": "test/database.js", "fullTitle": "Test db key methods"},
                    {"file": "test/meta.js", "fullTitle": "Meta functions"},
                ],
                "failures": [
                    {"file": "test/translator.js", "fullTitle": "Translator shim"},
                ],
                "pending": [],
            }
        )

        # Create a minimal parser.py that handles mocha JSON
        parser_script = """
import json
import sys
import dataclasses
from enum import Enum
from pathlib import Path
from typing import List

class TestStatus(Enum):
    PASSED = 1
    FAILED = 2
    SKIPPED = 3
    ERROR = 4

@dataclasses.dataclass
class TestResult:
    name: str
    status: TestStatus

def parse_test_output(stdout_content, stderr_content):
    results = []
    try:
        data = json.loads(stdout_content)
        for t in data.get("passes", []):
            results.append(TestResult(name=t.get("fullTitle", ""), status=TestStatus.PASSED))
        for t in data.get("failures", []):
            results.append(TestResult(name=t.get("fullTitle", ""), status=TestStatus.FAILED))
    except json.JSONDecodeError:
        pass
    return results

def export_to_json(results, output_path):
    json_results = {
        "tests": [
            {"name": r.name, "status": r.status.name} for r in results
        ]
    }
    with open(output_path, "w") as f:
        json.dump(json_results, f)

def main(stdout_path, stderr_path, output_path):
    with open(stdout_path) as f:
        stdout_content = f.read()
    with open(stderr_path) as f:
        stderr_content = f.read()
    results = parse_test_output(stdout_content, stderr_content)
    export_to_json(results, output_path)

if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
"""

        result = _parse_test_output_locally(parser_script, mocha_output, "", "test-instance")
        assert result.total == 3
        assert result.passed == 2

    def test_handles_parser_error(self) -> None:
        """Test that parser errors are handled gracefully."""
        bad_parser = "raise ValueError('broken')"
        result = _parse_test_output_locally(bad_parser, "output", "err", "test")
        assert result.total == 0
        assert result.passed == 0

    def test_handles_empty_output(self) -> None:
        """Test parsing with no test output."""
        parser_script = """
import json
import sys
from pathlib import Path

def main(stdout_path, stderr_path, output_path):
    with open(output_path, "w") as f:
        json.dump({"tests": []}, f)

if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
"""
        result = _parse_test_output_locally(parser_script, "", "", "test")
        assert result.total == 0
        assert result.passed == 0


class TestRunOfficialTests:
    """Tests for _run_official_tests orchestration."""

    @pytest.mark.asyncio
    async def test_runs_script_in_container(self) -> None:
        """Test that run_script.sh is written to container and executed."""
        env = MagicMock()
        env.uses_prebuilt = True
        env.write_file = AsyncMock()
        env.exec_command = AsyncMock(return_value=(0, '{"tests":[]}', ""))

        task = {
            "instance_id": "test-instance",
            "selected_test_files_to_run": '["test/foo.js", "test/bar.js"]',
        }

        # Use a simple parser that outputs empty test list
        parser = """
import json
import sys
from pathlib import Path

def main(stdout_path, stderr_path, output_path):
    with open(output_path, "w") as f:
        json.dump({"tests": []}, f)

if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
"""
        from mcpbr.benchmarks.swebench_pro import _run_official_tests

        await _run_official_tests(env, task, "#!/bin/bash\necho test", parser)

        # Verify run_script.sh was written to container
        env.write_file.assert_called_once_with(
            "run_script.sh", "#!/bin/bash\necho test", workdir="/app"
        )

        # Verify exec_command was called (chmod + the actual script run)
        assert env.exec_command.call_count == 2

    @pytest.mark.asyncio
    async def test_handles_no_selected_files(self) -> None:
        """Test graceful handling when no test files are specified."""
        env = MagicMock()
        env.uses_prebuilt = True

        task = {
            "instance_id": "test-instance",
            "selected_test_files_to_run": "[]",
        }

        from mcpbr.benchmarks.swebench_pro import _run_official_tests

        result = await _run_official_tests(env, task, "#!/bin/bash", "parser")
        assert result.total == 0
        assert result.passed == 0

    @pytest.mark.asyncio
    async def test_handles_timeout(self) -> None:
        """Test graceful handling when test execution times out."""
        env = MagicMock()
        env.uses_prebuilt = True
        env.write_file = AsyncMock()
        env.exec_command = AsyncMock(
            side_effect=[
                (0, "", ""),  # chmod succeeds
                TimeoutError("timed out"),  # script times out
            ]
        )

        task = {
            "instance_id": "test-instance",
            "selected_test_files_to_run": '["test/foo.js"]',
        }

        from mcpbr.benchmarks.swebench_pro import _run_official_tests

        result = await _run_official_tests(env, task, "#!/bin/bash", "parser")
        assert result.total == 0
        assert result.passed == 0


class TestSWEBenchProDockerImage:
    """Tests for pre-built image lookup."""

    def test_get_prebuilt_image_with_tag(self) -> None:
        task = {"dockerhub_tag": "django.django-django__django-abc123"}
        benchmark = SWEBenchProBenchmark()
        expected = f"{SWEBENCH_PRO_IMAGE_PREFIX}:django.django-django__django-abc123"
        assert benchmark.get_prebuilt_image(task) == expected

    def test_get_prebuilt_image_missing(self) -> None:
        task = {"instance_id": "test-123"}
        benchmark = SWEBenchProBenchmark()
        assert benchmark.get_prebuilt_image(task) is None


class TestSWEBenchProPromptTemplate:
    """Tests for prompt template."""

    def test_has_placeholder(self) -> None:
        benchmark = SWEBenchProBenchmark()
        template = benchmark.get_prompt_template()
        assert "{problem_statement}" in template

    def test_mentions_multiple_languages(self) -> None:
        benchmark = SWEBenchProBenchmark()
        template = benchmark.get_prompt_template()
        assert "Go" in template
        assert "TypeScript" in template
        assert "JavaScript" in template


class TestSWEBenchProFilterCategory:
    """Tests for category filtering in load_tasks."""

    @patch("mcpbr.benchmarks.swebench_pro.load_dataset")
    def test_filter_by_language(self, mock_load: MagicMock) -> None:
        mock_dataset = [
            {
                "instance_id": "t1",
                "repo": "django/django",
                "repo_language": "python",
                "problem_statement": "p",
                "base_commit": "c",
            },
            {
                "instance_id": "t2",
                "repo": "gin-gonic/gin",
                "repo_language": "go",
                "problem_statement": "p",
                "base_commit": "c",
            },
            {
                "instance_id": "t3",
                "repo": "vercel/next.js",
                "repo_language": "typescript",
                "problem_statement": "p",
                "base_commit": "c",
            },
        ]
        mock_load.return_value = MagicMock(
            __iter__=lambda self: iter(mock_dataset),
            __len__=lambda self: len(mock_dataset),
        )

        benchmark = SWEBenchProBenchmark()
        tasks = benchmark.load_tasks(filter_category=["go"])
        assert len(tasks) == 1
        assert tasks[0]["instance_id"] == "t2"

    @patch("mcpbr.benchmarks.swebench_pro.load_dataset")
    def test_filter_by_repo_substring(self, mock_load: MagicMock) -> None:
        mock_dataset = [
            {
                "instance_id": "t1",
                "repo": "django/django",
                "repo_language": "python",
                "problem_statement": "p",
                "base_commit": "c",
            },
            {
                "instance_id": "t2",
                "repo": "gin-gonic/gin",
                "repo_language": "go",
                "problem_statement": "p",
                "base_commit": "c",
            },
        ]
        mock_load.return_value = MagicMock(
            __iter__=lambda self: iter(mock_dataset),
            __len__=lambda self: len(mock_dataset),
        )

        benchmark = SWEBenchProBenchmark()
        tasks = benchmark.load_tasks(filter_category=["django"])
        assert len(tasks) == 1
        assert tasks[0]["instance_id"] == "t1"

    def test_pro_languages_set(self) -> None:
        assert {"python", "go", "typescript", "javascript", "ts", "js"} == PRO_LANGUAGES


class TestSWEBenchProLoadTasks:
    """Tests for task loading."""

    @patch("mcpbr.benchmarks.swebench_pro.load_dataset")
    def test_sample_size(self, mock_load: MagicMock) -> None:
        mock_dataset = [
            {
                "instance_id": f"t{i}",
                "repo": "r",
                "problem_statement": "p",
                "base_commit": "c",
            }
            for i in range(10)
        ]
        mock_ds = MagicMock()
        mock_ds.__iter__ = lambda self: iter(mock_dataset)
        mock_ds.__len__ = lambda self: len(mock_dataset)
        mock_ds.select = MagicMock(return_value=mock_dataset[:3])
        mock_load.return_value = mock_ds

        benchmark = SWEBenchProBenchmark()
        tasks = benchmark.load_tasks(sample_size=3)
        assert len(tasks) == 3

    @patch("mcpbr.benchmarks.swebench_pro.load_dataset")
    def test_task_ids(self, mock_load: MagicMock) -> None:
        mock_dataset = [
            {
                "instance_id": f"t{i}",
                "repo": "r",
                "problem_statement": "p",
                "base_commit": "c",
            }
            for i in range(5)
        ]
        mock_load.return_value = MagicMock(
            __iter__=lambda self: iter(mock_dataset),
            __len__=lambda self: len(mock_dataset),
        )

        benchmark = SWEBenchProBenchmark()
        tasks = benchmark.load_tasks(task_ids=["t1", "t3"])
        assert len(tasks) == 2
        ids = {t["instance_id"] for t in tasks}
        assert ids == {"t1", "t3"}

    @patch("mcpbr.benchmarks.swebench_pro.load_dataset")
    def test_combined_filters(self, mock_load: MagicMock) -> None:
        mock_dataset = [
            {
                "instance_id": "t1",
                "repo": "django/django",
                "repo_language": "python",
                "problem_statement": "p",
                "base_commit": "c",
            },
            {
                "instance_id": "t2",
                "repo": "gin-gonic/gin",
                "repo_language": "go",
                "problem_statement": "p",
                "base_commit": "c",
            },
            {
                "instance_id": "t3",
                "repo": "vercel/next.js",
                "repo_language": "typescript",
                "problem_statement": "p",
                "base_commit": "c",
            },
        ]
        mock_load.return_value = MagicMock(
            __iter__=lambda self: iter(mock_dataset),
            __len__=lambda self: len(mock_dataset),
        )

        benchmark = SWEBenchProBenchmark()
        tasks = benchmark.load_tasks(
            task_ids=["t1", "t2"],
            filter_category=["python"],
        )
        assert len(tasks) == 1
        assert tasks[0]["instance_id"] == "t1"


class TestSWEBenchProSandboxLevel:
    """Tests for sandbox level."""

    def test_default_sandbox_level(self) -> None:
        benchmark = SWEBenchProBenchmark()
        assert benchmark.get_default_sandbox_level() is None


class TestSWEBenchProRegistry:
    """Tests for benchmark registry integration."""

    def test_create_swebench_pro(self) -> None:
        from mcpbr.benchmarks import create_benchmark

        benchmark = create_benchmark("swe-bench-pro")
        assert isinstance(benchmark, SWEBenchProBenchmark)
        assert benchmark.dataset == "ScaleAI/SWE-bench_Pro"

    def test_listed_in_registry(self) -> None:
        from mcpbr.benchmarks import list_benchmarks

        assert "swe-bench-pro" in list_benchmarks()


class TestSWEBenchProEvalResultToDict:
    """Tests for _eval_result_to_dict helper."""

    def test_basic_conversion(self) -> None:
        from mcpbr.evaluation import EvaluationResult

        benchmark = SWEBenchProBenchmark()
        result = EvaluationResult(
            resolved=True,
            patch_applied=True,
            fail_to_pass=TestResults(passed=2, total=2, details=[]),
            pass_to_pass=TestResults(passed=5, total=5, details=[]),
        )
        d = benchmark._eval_result_to_dict(result)
        assert d["resolved"] is True
        assert d["patch_applied"] is True
        assert d["fail_to_pass"]["passed"] == 2
        assert d["pass_to_pass"]["passed"] == 5

    def test_with_error(self) -> None:
        from mcpbr.evaluation import EvaluationResult

        benchmark = SWEBenchProBenchmark()
        result = EvaluationResult(
            resolved=False,
            patch_applied=False,
            error="Patch failed",
        )
        d = benchmark._eval_result_to_dict(result)
        assert d["resolved"] is False
        assert d["eval_error"] == "Patch failed"
        assert "fail_to_pass" not in d
