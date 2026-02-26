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

import logging
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

logger = logging.getLogger(__name__)

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

        # Detect JS/TS test runner once (avoids repeated detection per test)
        js_runner = "jest"
        if language in ("typescript", "javascript", "ts", "js"):
            js_runner = await _detect_js_runner(env, workdir=workdir)

        results = []
        passed = 0

        for test in tests:
            # SWE-bench Pro images don't use conda — never prepend conda activation
            # for non-Python languages (uses_prebuilt=False disables it)
            test_cmd = _build_pro_test_command(
                test, language, uses_prebuilt=False, js_runner=js_runner
            )
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


_KNOWN_RUNNERS = ("jest", "mocha", "vitest", "ospec", "ava")


async def _detect_js_runner(env: "TaskEnvironment", workdir: str | None = None) -> str:
    """Detect the JavaScript/TypeScript test runner installed in a container.

    Detection strategy:
    1. Check node_modules/.bin/ for known runner binaries
    2. Parse package.json scripts.test for runner hints
    3. Fall back to "npm" (runs npm test) if nothing is detected

    Args:
        env: Task environment with exec_command.
        workdir: Working directory inside container.

    Returns:
        Runner name: "jest", "mocha", "vitest", "ospec", "ava", or "npm".
    """
    # Check for runner binaries in node_modules
    detect_cmd = (
        "if [ -f node_modules/.bin/jest ]; then echo jest; "
        "elif [ -f node_modules/.bin/mocha ]; then echo mocha; "
        "elif [ -f node_modules/.bin/vitest ]; then echo vitest; "
        "elif [ -f node_modules/.bin/ospec ]; then echo ospec; "
        "elif [ -f node_modules/.bin/ava ]; then echo ava; "
        "else echo none; fi"
    )
    try:
        exit_code, stdout, _ = await env.exec_command(detect_cmd, timeout=10, workdir=workdir)
        if exit_code == 0 and stdout:
            runner = stdout.strip().split("\n")[-1].strip()
            if runner in _KNOWN_RUNNERS:
                return runner
    except Exception:
        logger.debug("Failed to detect JS test runner from node_modules")

    # Fallback: parse package.json scripts.test for runner hints
    pkg_cmd = (
        "node -e \"try{const p=require('./package.json');"
        "console.log(p.scripts&&p.scripts.test||'')}catch(e){console.log('')}\" 2>/dev/null"
    )
    try:
        exit_code, stdout, _ = await env.exec_command(pkg_cmd, timeout=10, workdir=workdir)
        if exit_code == 0 and stdout:
            test_script = stdout.strip().split("\n")[-1].strip().lower()
            for runner in _KNOWN_RUNNERS:
                if runner in test_script:
                    return runner
    except Exception:
        logger.debug("Failed to detect JS test runner from package.json")

    # Ultimate fallback: use npm test
    return "npm"


def _build_pro_test_command(
    test: str,
    language: str,
    uses_prebuilt: bool = False,
    js_runner: str = "jest",
) -> str:
    """Build a language-specific test command for SWE-bench Pro.

    Test ID formats by language:
        Go: "TestFoo", "TestFoo/subtest", "TestFoo/#00"
        JS/TS: "file.js | test description", "file.ts | suite name"
        Python: "tests/test_foo.py::TestClass::test_method"

    Args:
        test: Test identifier.
        language: Programming language (python, go, typescript, javascript, js, ts).
        uses_prebuilt: Whether a pre-built image is being used (adds conda activation).
        js_runner: JavaScript test runner ("jest", "mocha", or "vitest").

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
        # Go test IDs are always function names, optionally with subtests via /
        # e.g., "TestFoo", "TestFoo/subtest", "TestFoo/#00", "TestFoo//api/v1"
        # Always use -run with the top-level test name and ./... to search all packages
        if "/" in test:
            # Extract top-level test name (before first /)
            top_level = test.split("/", 1)[0]
            return f"{activate}go test -v -count=1 -run {shlex.quote(top_level)} ./... 2>&1"
        else:
            return f"{activate}go test -v -count=1 -run {shlex.quote(test)} ./... 2>&1"

    if language in ("typescript", "javascript", "ts", "js"):
        return _build_js_test_command(test, js_runner, activate)

    # Fallback: try running as-is
    return f"{activate}{test} 2>&1"


def _build_js_test_command(test: str, runner: str, activate: str = "") -> str:
    """Build a JS/TS test command for the detected runner.

    Args:
        test: Test identifier in "file | description" format.
        runner: Test runner name ("jest", "mocha", "vitest", "ospec", "ava", "npm").
        activate: Optional conda activation prefix.

    Returns:
        Shell command string.
    """
    import shlex

    # Parse "file | description" format
    file_path = ""
    test_name = ""
    if " | " in test:
        parts = test.split(" | ", 1)
        file_path = parts[0].strip()
        test_name = parts[1].strip()
    elif "/" in test or test.endswith((".ts", ".js", ".tsx", ".jsx")):
        file_path = test
    else:
        test_name = test

    if runner == "mocha":
        cmd = f"{activate}npx mocha"
        if file_path:
            cmd += f" {shlex.quote(file_path)}"
        if test_name and test_name != "test suite":
            cmd += f" --grep {shlex.quote(test_name)}"
        cmd += " --timeout 30000 2>&1"
        return cmd

    if runner == "vitest":
        cmd = f"{activate}npx vitest run"
        if file_path:
            cmd += f" {shlex.quote(file_path)}"
        if test_name and test_name != "test suite":
            cmd += f" -t {shlex.quote(test_name)}"
        cmd += " 2>&1"
        return cmd

    if runner == "ospec":
        # ospec: run file directly with node (ospec tests are self-executing)
        if file_path:
            cmd = f"{activate}node {shlex.quote(file_path)} 2>&1"
        else:
            cmd = f"{activate}npx ospec 2>&1"
        return cmd

    if runner == "ava":
        cmd = f"{activate}npx ava"
        if file_path:
            cmd += f" {shlex.quote(file_path)}"
        if test_name and test_name != "test suite":
            cmd += f" -m {shlex.quote(test_name)}"
        cmd += " 2>&1"
        return cmd

    if runner == "npm":
        # Fallback: use npm test, passing file as argument if possible
        if file_path:
            cmd = f"{activate}npm test -- {shlex.quote(file_path)} 2>&1"
        elif test_name:
            cmd = f"{activate}npm test 2>&1"
        else:
            cmd = f"{activate}npm test 2>&1"
        return cmd

    # Default: jest
    cmd = f"{activate}npx jest"
    if file_path:
        cmd += f" {shlex.quote(file_path)}"
    if test_name and test_name != "test suite":
        cmd += f" -t {shlex.quote(test_name)}"
    cmd += " --verbose --no-cache 2>&1"
    return cmd
