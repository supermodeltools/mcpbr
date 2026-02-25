"""SWE-bench Pro benchmark implementation.

SWE-bench Pro is a multi-language benchmark with 731 instances across 11 repos
in Python, Go, TypeScript, and JavaScript. Average solutions span 107.4 lines
across 4.1 files. Top models achieve ~23% resolution (vs 70%+ on Verified).

Key differences from SWE-bench:
- Docker images from DockerHub (dockerhub_tag field) instead of GHCR
- Multi-language test runners (Python, Go, TypeScript, JavaScript)
- Lowercase field names (fail_to_pass instead of FAIL_TO_PASS)
- Language metadata per task (repo_language field)
"""

from typing import Any

from datasets import load_dataset

from ..docker_env import DockerEnvironmentManager, TaskEnvironment
from ..evaluation import (
    EvaluationResult,
    evaluate_patch,
    get_test_list_field,
    parse_test_list,
    run_tests,
)
from .base import BenchmarkTask

# Supported languages in SWE-bench Pro
PRO_LANGUAGES = {"python", "go", "typescript", "javascript", "ts", "js"}

# Aliases: user-friendly names → dataset values
_LANGUAGE_ALIASES: dict[str, str] = {
    "javascript": "js",
    "typescript": "ts",
}

# DockerHub registry prefix for SWE-bench Pro pre-built images
SWEBENCH_PRO_IMAGE_PREFIX = "jefzda/sweap-images"


class SWEBenchProBenchmark:
    """SWE-bench Pro benchmark implementation.

    Multi-language benchmark for evaluating coding agents on real-world
    software engineering tasks across Python, Go, TypeScript, and JavaScript.
    """

    name = "swe-bench-pro"

    def __init__(self, dataset: str = "ScaleAI/SWE-bench_Pro"):
        """Initialize SWE-bench Pro benchmark.

        Args:
            dataset: HuggingFace dataset identifier.
        """
        self.dataset = dataset

    def load_tasks(
        self,
        sample_size: int | None = None,
        task_ids: list[str] | None = None,
        level: int | None = None,
        filter_difficulty: list[str] | None = None,
        filter_category: list[str] | None = None,
        filter_tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Load tasks from SWE-bench Pro dataset.

        Args:
            sample_size: Maximum number of tasks to load (None for all).
            task_ids: Specific task IDs to load (None for all).
            level: Unused for SWE-bench Pro.
            filter_difficulty: Unused for SWE-bench Pro.
            filter_category: Filter by language name (e.g., "python", "go")
                or repository substring (e.g., "django", "gin-gonic").
            filter_tags: Unused for SWE-bench Pro.

        Returns:
            List of SWE-bench Pro task dictionaries.
        """
        dataset = load_dataset(self.dataset, split="test")

        # Optimization: early truncation when no filtering is needed
        needs_full_scan = bool(task_ids) or bool(filter_category)
        if not needs_full_scan and sample_size is not None and len(dataset) > sample_size:
            dataset = dataset.select(range(sample_size))

        if task_ids:
            task_id_set = set(task_ids)
            tasks = [item for item in dataset if item["instance_id"] in task_id_set]
        else:
            tasks = list(dataset)

        if filter_category:
            filtered = []
            for task in tasks:
                repo = task.get("repo", "")
                language = task.get("repo_language", "").lower()
                for category in filter_category:
                    cat_lower = category.lower()
                    # If the category is a known language, match by language only
                    if cat_lower in PRO_LANGUAGES:
                        # Resolve aliases (e.g., "javascript" -> "js")
                        resolved = _LANGUAGE_ALIASES.get(cat_lower, cat_lower)
                        if resolved == language:
                            filtered.append(task)
                            break
                    elif cat_lower in repo.lower():
                        # Otherwise, match by repo substring
                        filtered.append(task)
                        break
            tasks = filtered

        if sample_size is not None and len(tasks) > sample_size:
            tasks = tasks[:sample_size]

        return tasks

    def normalize_task(self, task: dict[str, Any]) -> BenchmarkTask:
        """Convert SWE-bench Pro task to normalized format.

        Handles both lowercase (SWE-bench Pro) and uppercase (SWE-bench)
        field names for test lists.

        Args:
            task: SWE-bench Pro task dictionary.

        Returns:
            Normalized BenchmarkTask.
        """
        return BenchmarkTask(
            task_id=task["instance_id"],
            problem_statement=task["problem_statement"],
            repo=task["repo"],
            commit=task["base_commit"],
            metadata={
                "fail_to_pass": get_test_list_field(task, "fail_to_pass"),
                "pass_to_pass": get_test_list_field(task, "pass_to_pass"),
                "test_patch": task.get("test_patch", ""),
                "repo_language": task.get("repo_language", "unknown"),
            },
        )

    async def create_environment(
        self,
        task: dict[str, Any],
        docker_manager: DockerEnvironmentManager,
    ) -> TaskEnvironment:
        """Create environment for SWE-bench Pro task.

        Injects the DockerHub image override so DockerEnvironmentManager
        pulls from DockerHub instead of GHCR.

        Args:
            task: SWE-bench Pro task dictionary.
            docker_manager: Docker environment manager.

        Returns:
            TaskEnvironment for the task.
        """
        # Inject image override for DockerHub-hosted images
        # The dockerhub_tag field is the tag portion; prepend the registry prefix
        # SWE-bench Pro images use /app as workdir (not /testbed)
        task_copy = dict(task)
        dockerhub_tag = task.get("dockerhub_tag")
        if dockerhub_tag:
            task_copy["_image_override"] = f"{SWEBENCH_PRO_IMAGE_PREFIX}:{dockerhub_tag}"
            task_copy["_workdir_override"] = "/app"

        return await docker_manager.create_environment(task_copy)

    async def evaluate(
        self,
        env: TaskEnvironment,
        task: dict[str, Any],
        solution: str,
    ) -> dict[str, Any]:
        """Evaluate a patch for SWE-bench Pro task.

        For Python tasks, delegates to the existing evaluate_patch().
        For Go/TypeScript/JavaScript, uses language-specific test runners.

        Args:
            env: Task environment.
            task: SWE-bench Pro task dictionary.
            solution: Unified diff patch to evaluate.

        Returns:
            Dictionary with evaluation results including 'resolved' boolean.
        """
        language = task.get("repo_language", "python").lower()

        if language == "python":
            # Delegate Python evaluation to existing logic
            eval_result: EvaluationResult = await evaluate_patch(env, task, solution)
            return self._eval_result_to_dict(eval_result)

        # For non-Python languages, use language-specific evaluation
        return await self._evaluate_multilang(env, task, solution, language)

    async def _evaluate_multilang(
        self,
        env: TaskEnvironment,
        task: dict[str, Any],
        patch: str,
        language: str,
    ) -> dict[str, Any]:
        """Evaluate a patch using language-specific test runners.

        Args:
            env: Task environment.
            task: SWE-bench Pro task dictionary.
            patch: Unified diff patch to evaluate.
            language: Programming language (go, typescript, javascript).

        Returns:
            Dictionary with evaluation results.
        """
        from ..evaluation import _apply_test_patch, apply_patch

        # SWE-bench Pro images use /app as their working directory
        eval_workdir = "/app" if env.uses_prebuilt else None

        applied, error = await apply_patch(env, patch, workdir=eval_workdir)
        if not applied:
            return {"resolved": False, "patch_applied": False, "eval_error": error}

        test_patch = task.get("test_patch", "")
        if test_patch:
            await _apply_test_patch(env, test_patch, workdir=eval_workdir)

        # Reinstall package so patched code is active (SWE-bench Pro images
        # install into site-packages, not editable mode)
        if eval_workdir and language == "python":
            await env.exec_command(
                "pip install -e . -q 2>/dev/null || true",
                timeout=120,
                workdir=eval_workdir,
            )

        fail_to_pass_str = get_test_list_field(task, "fail_to_pass")
        pass_to_pass_str = get_test_list_field(task, "pass_to_pass")
        fail_to_pass_tests = parse_test_list(fail_to_pass_str)
        pass_to_pass_tests = parse_test_list(pass_to_pass_str)

        fail_to_pass_results = await self._run_lang_tests(
            env, fail_to_pass_tests, language, workdir=eval_workdir
        )
        pass_to_pass_results = await self._run_lang_tests(
            env, pass_to_pass_tests[:10], language, workdir=eval_workdir
        )

        resolved = (
            fail_to_pass_results.passed == fail_to_pass_results.total
            and fail_to_pass_results.total > 0
            and pass_to_pass_results.passed == pass_to_pass_results.total
        )

        result: dict[str, Any] = {"resolved": resolved, "patch_applied": True}
        if fail_to_pass_results:
            result["fail_to_pass"] = {
                "passed": fail_to_pass_results.passed,
                "total": fail_to_pass_results.total,
            }
        if pass_to_pass_results:
            result["pass_to_pass"] = {
                "passed": pass_to_pass_results.passed,
                "total": pass_to_pass_results.total,
            }
        return result

    async def _run_lang_tests(
        self,
        env: TaskEnvironment,
        tests: list[str],
        language: str,
        workdir: str | None = None,
        timeout: int = 120,
    ) -> Any:
        """Run tests using language-specific commands.

        Args:
            env: Task environment.
            tests: List of test identifiers.
            language: Programming language.
            workdir: Working directory.
            timeout: Timeout per test in seconds.

        Returns:
            TestResults instance.
        """
        if language == "python":
            return await run_tests(
                env, tests, timeout=timeout, uses_prebuilt=env.uses_prebuilt, workdir=workdir
            )

        # For non-Python, build language-specific commands and run
        from ..evaluation import TestResults

        if not tests:
            return TestResults(passed=0, total=0, details=[])

        results = []
        passed = 0

        for test in tests:
            # SWE-bench Pro images don't use conda — never prepend conda activation
            # for non-Python languages (uses_prebuilt=False disables it)
            test_cmd = _build_pro_test_command(test, language, uses_prebuilt=False)
            try:
                exit_code, stdout, stderr = await env.exec_command(
                    test_cmd, timeout=timeout, workdir=workdir
                )
                test_passed = exit_code == 0
                if test_passed:
                    passed += 1
                results.append(
                    {
                        "test": test,
                        "passed": test_passed,
                        "exit_code": exit_code,
                        "output": stdout[:1000] if stdout else "",
                        "error": stderr[:1000] if stderr else "",
                    }
                )
            except TimeoutError:
                results.append(
                    {
                        "test": test,
                        "passed": False,
                        "exit_code": -1,
                        "output": "",
                        "error": "Test timed out",
                    }
                )

        return TestResults(passed=passed, total=len(tests), details=results)

    def _eval_result_to_dict(self, eval_result: EvaluationResult) -> dict[str, Any]:
        """Convert EvaluationResult to dictionary format."""
        result: dict[str, Any] = {
            "resolved": eval_result.resolved,
            "patch_applied": eval_result.patch_applied,
        }
        if eval_result.fail_to_pass:
            result["fail_to_pass"] = {
                "passed": eval_result.fail_to_pass.passed,
                "total": eval_result.fail_to_pass.total,
            }
        if eval_result.pass_to_pass:
            result["pass_to_pass"] = {
                "passed": eval_result.pass_to_pass.passed,
                "total": eval_result.pass_to_pass.total,
            }
        if eval_result.error:
            result["eval_error"] = eval_result.error
        return result

    def get_prebuilt_image(self, task: dict[str, Any]) -> str | None:
        """Get pre-built Docker image name for the task.

        SWE-bench Pro uses DockerHub images specified in the dockerhub_tag field.

        Args:
            task: SWE-bench Pro task dictionary.

        Returns:
            Full DockerHub image name, or None if not available.
        """
        tag = task.get("dockerhub_tag")
        if tag:
            return f"{SWEBENCH_PRO_IMAGE_PREFIX}:{tag}"
        return None

    def get_prompt_template(self) -> str:
        """Get SWE-bench Pro prompt template.

        Returns:
            Prompt template for fixing bugs across multiple languages.
        """
        return (
            "Fix the following bug in this repository:\n\n"
            "{problem_statement}\n\n"
            "IMPORTANT CONSTRAINTS:\n"
            "- Only modify the minimum files necessary to fix the bug\n"
            "- Do NOT create new test files\n"
            "- Do NOT create documentation files\n"
            "- Do NOT create reproduction scripts\n"
            "- Focus solely on the fix in existing source files\n"
            "- This may be a Python, Go, TypeScript, or JavaScript project"
        )

    def get_default_sandbox_level(self) -> str | None:
        """Get default sandbox level for SWE-bench Pro."""
        return None


def _build_pro_test_command(test: str, language: str, uses_prebuilt: bool = False) -> str:
    """Build a language-specific test command for SWE-bench Pro.

    Args:
        test: Test identifier.
        language: Programming language (python, go, typescript, javascript).
        uses_prebuilt: Whether a pre-built image is being used.

    Returns:
        Shell command string to run the test.
    """
    import shlex

    from ..evaluation import _build_test_command, _normalize_test_id

    if language == "python":
        return _build_test_command(test, uses_prebuilt)

    test = _normalize_test_id(test)

    if uses_prebuilt:
        activate = "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed && "
    else:
        activate = ""

    if language == "go":
        # Go test identifiers can be package paths or test function names
        if "/" in test or test.startswith("."):
            # Package path: go test -v ./path/to/package
            return f"{activate}go test -v -count=1 {shlex.quote(test)} 2>&1"
        else:
            # Test function name: go test -v -run TestName ./...
            return f"{activate}go test -v -count=1 -run {shlex.quote(test)} ./... 2>&1"

    if language in ("typescript", "javascript"):
        # Jest-style test identifiers
        if "/" in test or test.endswith((".ts", ".js", ".tsx", ".jsx")):
            # File path
            return f"{activate}npx jest {shlex.quote(test)} --verbose --no-cache 2>&1"
        else:
            # Test name pattern
            return f"{activate}npx jest -t {shlex.quote(test)} --verbose --no-cache 2>&1"

    # Fallback: try running as-is
    return f"{activate}{test} 2>&1"
