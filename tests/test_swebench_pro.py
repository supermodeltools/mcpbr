"""Tests for SWE-bench Pro benchmark implementation."""

from unittest.mock import MagicMock, patch

from mcpbr.benchmarks.swebench_pro import (
    PRO_LANGUAGES,
    SWEBENCH_PRO_IMAGE_PREFIX,
    SWEBenchProBenchmark,
    _build_pro_test_command,
)


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


class TestBuildProTestCommand:
    """Tests for language-specific test command building."""

    def test_python_delegates(self) -> None:
        """Python should delegate to existing _build_test_command."""
        cmd = _build_pro_test_command("tests/test_foo.py::test_bar", "python")
        assert "pytest" in cmd or "test_foo" in cmd

    def test_go_function_name(self) -> None:
        cmd = _build_pro_test_command("TestRouteMatching", "go")
        assert "go test" in cmd
        assert "-run" in cmd
        assert "TestRouteMatching" in cmd
        assert "./..." in cmd

    def test_go_subtest(self) -> None:
        """Go subtests (TestFoo/#00, TestFoo/subtest) use top-level name with -run."""
        cmd = _build_pro_test_command("TestParseResourcePath/#00", "go")
        assert "go test" in cmd
        assert "-run" in cmd
        assert "TestParseResourcePath" in cmd
        assert "./..." in cmd

    def test_typescript_file(self) -> None:
        cmd = _build_pro_test_command("src/__tests__/parser.test.ts", "typescript")
        assert "npx jest" in cmd
        assert "parser.test.ts" in cmd

    def test_typescript_pattern(self) -> None:
        cmd = _build_pro_test_command("should parse tokens", "typescript")
        assert "npx jest" in cmd
        assert "-t" in cmd

    def test_javascript_file(self) -> None:
        cmd = _build_pro_test_command("test/index.test.js", "javascript")
        assert "npx jest" in cmd
        assert "index.test.js" in cmd

    def test_javascript_pattern(self) -> None:
        cmd = _build_pro_test_command("handles edge case", "javascript")
        assert "npx jest" in cmd

    def test_js_pipe_format(self) -> None:
        """SWE-bench Pro JS format: 'file.js | test description'."""
        cmd = _build_pro_test_command("test/database.js | Test database key methods", "js")
        assert "npx jest" in cmd
        assert "test/database.js" in cmd
        assert "-t" in cmd
        assert "Test database key methods" in cmd

    def test_ts_test_suite_format(self) -> None:
        """TS 'test suite' format runs the whole file without -t filter."""
        cmd = _build_pro_test_command("test/tests/LoginFacadeTest.js | test suite", "ts")
        assert "npx jest" in cmd
        assert "test/tests/LoginFacadeTest.js" in cmd
        assert "-t" not in cmd

    def test_prebuilt_conda_activation(self) -> None:
        cmd = _build_pro_test_command("TestFoo", "go", uses_prebuilt=True)
        assert "conda activate testbed" in cmd

    def test_unknown_language_fallback(self) -> None:
        cmd = _build_pro_test_command("test_something", "rust")
        assert "test_something" in cmd


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
        from mcpbr.evaluation import EvaluationResult, TestResults

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
