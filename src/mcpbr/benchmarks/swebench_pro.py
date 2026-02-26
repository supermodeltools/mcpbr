"""SWE-bench Pro benchmark implementation.

SWE-bench Pro is a multi-language benchmark with 731 instances across 11 repos
in Python, Go, TypeScript, and JavaScript. Average solutions span 107.4 lines
across 4.1 files. Top models achieve ~23% resolution (vs 70%+ on Verified).

Key differences from SWE-bench:
- Docker images from DockerHub (dockerhub_tag field) instead of GHCR
- Multi-language test runners (Python, Go, TypeScript, JavaScript)
- Lowercase field names (fail_to_pass instead of FAIL_TO_PASS)
- Language metadata per task (repo_language field)

Test execution uses official run scripts from scaleapi/SWE-bench_Pro-os,
which handle per-repo test infrastructure (e.g., Redis for NodeBB,
ansible-test for ansible, custom runners for tutanota).
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from datasets import load_dataset

from ..docker_env import DockerEnvironmentManager, TaskEnvironment
from ..evaluation import (
    EvaluationResult,
    TestResults,
    _apply_test_patch,
    apply_patch,
    evaluate_patch,
    get_test_list_field,
    parse_test_list,
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

# Git URL for the official SWE-bench Pro run scripts repository
_RUN_SCRIPTS_REPO = "https://github.com/scaleapi/SWE-bench_Pro-os.git"

# Default cache directory for cloned run scripts
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "mcpbr" / "swebench-pro-scripts"


def _ensure_run_scripts_repo(cache_dir: Path | None = None) -> Path:
    """Clone or update the official SWE-bench Pro run scripts repository.

    Performs a shallow clone of scaleapi/SWE-bench_Pro-os into the cache
    directory. If the repo already exists, reuses it.

    Args:
        cache_dir: Directory to clone into. Defaults to ~/.cache/mcpbr/swebench-pro-scripts/.

    Returns:
        Path to the cloned repository root.
    """
    repo_dir = cache_dir or _DEFAULT_CACHE_DIR

    if (repo_dir / "run_scripts").is_dir():
        logger.debug("Run scripts repo already cached at %s", repo_dir)
        return repo_dir

    repo_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Cloning SWE-bench Pro run scripts to %s", repo_dir)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            _RUN_SCRIPTS_REPO,
            str(repo_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    # Sparse checkout only the run_scripts directory
    subprocess.run(
        ["git", "sparse-checkout", "set", "run_scripts"],
        cwd=str(repo_dir),
        check=True,
        capture_output=True,
        text=True,
    )

    return repo_dir


def _get_instance_scripts(repo_path: Path, instance_id: str) -> tuple[str, str]:
    """Read the run_script.sh and parser.py for a specific instance.

    Args:
        repo_path: Path to the cloned SWE-bench_Pro-os repository.
        instance_id: Instance ID matching the directory name in run_scripts/.

    Returns:
        Tuple of (run_script_content, parser_content).

    Raises:
        FileNotFoundError: If instance scripts don't exist.
    """
    instance_dir = repo_path / "run_scripts" / instance_id

    run_script_path = instance_dir / "run_script.sh"
    parser_path = instance_dir / "parser.py"

    if not run_script_path.exists():
        raise FileNotFoundError(
            f"No run_script.sh found for instance {instance_id} at {run_script_path}"
        )
    if not parser_path.exists():
        raise FileNotFoundError(f"No parser.py found for instance {instance_id} at {parser_path}")

    return run_script_path.read_text(), parser_path.read_text()


async def _run_before_repo_set_cmd(
    env: TaskEnvironment,
    task: dict[str, Any],
    workdir: str | None = None,
) -> None:
    """Run the before_repo_set_cmd from the dataset after patch application.

    The official SWE-bench Pro evaluation harness runs the last line of
    before_repo_set_cmd between applying the patch and running tests.
    This typically restores specific test files from the fix commit, e.g.:
        git checkout <commit> -- test/tests/SomeTest.ts

    The earlier lines (git reset, git clean, git checkout <base>) are
    redundant because our apply_patch() already handles that.

    Args:
        env: Task environment.
        task: SWE-bench Pro task dictionary.
        workdir: Working directory inside container.
    """
    before_cmd = task.get("before_repo_set_cmd", "")
    if not before_cmd or not before_cmd.strip():
        return

    # The official harness only uses the last line
    last_line = before_cmd.strip().split("\n")[-1].strip()
    if not last_line:
        return

    # Skip if it's just a git reset/clean/checkout <hash> (already done by apply_patch)
    # We only care about "git checkout <hash> -- <file>" which restores specific files
    if last_line.startswith("git checkout") and " -- " not in last_line:
        return
    if last_line.startswith(("git reset", "git clean")):
        return

    logger.debug("Running before_repo_set_cmd for %s: %s", task.get("instance_id"), last_line)
    try:
        await env.exec_command(last_line, timeout=60, workdir=workdir)
    except Exception:
        logger.warning(
            "before_repo_set_cmd failed for %s: %s",
            task.get("instance_id"),
            last_line,
        )


async def _run_official_tests(
    env: TaskEnvironment,
    task: dict[str, Any],
    run_script: str,
    parser_script: str,
    timeout: int = 300,
) -> TestResults:
    """Run tests using the official SWE-bench Pro run scripts.

    Copies run_script.sh into the container, executes it with the selected
    test files, captures stdout/stderr, then runs parser.py locally on
    the host to parse results.

    Args:
        env: Task environment with a running container.
        task: SWE-bench Pro task dictionary (needs selected_test_files_to_run).
        run_script: Content of run_script.sh.
        parser_script: Content of parser.py.
        timeout: Timeout for test execution in seconds.

    Returns:
        TestResults with parsed pass/fail counts.
    """
    eval_workdir = "/app" if env.uses_prebuilt else None

    # Build test files argument from selected_test_files_to_run
    selected_files_raw = task.get("selected_test_files_to_run", "[]")
    try:
        selected_files = (
            json.loads(selected_files_raw)
            if isinstance(selected_files_raw, str)
            else selected_files_raw
        )
    except (json.JSONDecodeError, TypeError):
        selected_files = []

    if not selected_files:
        logger.warning("No selected_test_files_to_run for %s", task.get("instance_id"))
        return TestResults(passed=0, total=0, details=[])

    # Write run_script.sh to container
    await env.write_file("run_script.sh", run_script, workdir=eval_workdir)
    await env.exec_command("chmod +x /app/run_script.sh", timeout=10, workdir=eval_workdir)

    # Join test files as comma-separated argument
    test_files_arg = ",".join(str(f) for f in selected_files)

    # Run the official test script
    try:
        _exit_code, stdout, stderr = await env.exec_command(
            f"bash /app/run_script.sh '{test_files_arg}'",
            timeout=timeout,
            workdir=eval_workdir,
        )
    except TimeoutError:
        logger.warning("Test execution timed out for %s", task.get("instance_id"))
        return TestResults(passed=0, total=0, details=[{"error": "Test timed out"}])

    # Run parser.py locally on host to parse the test output
    return _parse_test_output_locally(
        parser_script, stdout, stderr, task.get("instance_id", "unknown")
    )


def _parse_test_output_locally(
    parser_script: str,
    stdout: str,
    stderr: str,
    instance_id: str,
) -> TestResults:
    """Run parser.py as a local subprocess to parse test output.

    The parser runs on the host (not in the container) because Go/JS/TS
    container images may not have Python installed.

    Args:
        parser_script: Content of parser.py.
        stdout: Captured stdout from test execution.
        stderr: Captured stderr from test execution.
        instance_id: Instance ID for logging.

    Returns:
        TestResults parsed from the output.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        parser_path = tmp / "parser.py"
        stdout_path = tmp / "stdout.log"
        stderr_path = tmp / "stderr.log"
        output_path = tmp / "output.json"

        parser_path.write_text(parser_script)
        stdout_path.write_text(stdout or "")
        stderr_path.write_text(stderr or "")

        try:
            result = subprocess.run(
                [
                    "python3",
                    str(parser_path),
                    str(stdout_path),
                    str(stderr_path),
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Parser timed out for %s", instance_id)
            return TestResults(passed=0, total=0, details=[{"error": "Parser timed out"}])

        if result.returncode != 0:
            logger.warning("Parser failed for %s: %s", instance_id, result.stderr[:500])
            return TestResults(
                passed=0,
                total=0,
                details=[{"error": f"Parser failed: {result.stderr[:500]}"}],
            )

        if not output_path.exists():
            logger.warning("Parser produced no output.json for %s", instance_id)
            return TestResults(passed=0, total=0, details=[])

        try:
            output_data = json.loads(output_path.read_text())
        except json.JSONDecodeError:
            logger.warning("Parser output is not valid JSON for %s", instance_id)
            return TestResults(passed=0, total=0, details=[])

        tests = output_data.get("tests", [])
        passed = sum(1 for t in tests if t.get("status") == "PASSED")
        total = len(tests)

        details = [
            {
                "test": t.get("name", "unknown"),
                "passed": t.get("status") == "PASSED",
                "status": t.get("status", "UNKNOWN"),
            }
            for t in tests
        ]

        return TestResults(passed=passed, total=total, details=details)


def _match_test_results(
    parsed_results: TestResults,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
) -> tuple[TestResults, TestResults]:
    """Match parsed test results against expected fail_to_pass and pass_to_pass lists.

    The parser produces test names like "file.js | test description" or
    "TestFoo" etc. We match these against the expected test lists from
    the dataset.

    Args:
        parsed_results: TestResults from _run_official_tests.
        fail_to_pass: Expected tests that should pass (were failing before fix).
        pass_to_pass: Expected tests that should still pass (regression check).

    Returns:
        Tuple of (fail_to_pass_results, pass_to_pass_results).
    """
    # Build a lookup of parsed test name → status
    parsed_status: dict[str, str] = {}
    for detail in parsed_results.details:
        name = detail.get("test", "")
        status = detail.get("status", "UNKNOWN")
        if name:
            parsed_status[name] = status

    def _check_tests(expected: list[str]) -> TestResults:
        if not expected:
            return TestResults(passed=0, total=0, details=[])

        passed = 0
        details = []
        for test_name in expected:
            # Try exact match first
            status = parsed_status.get(test_name)

            # If no exact match, try substring matching (parser may add
            # file prefixes or slightly different formatting)
            if status is None:
                for parsed_name, parsed_stat in parsed_status.items():
                    if test_name in parsed_name or parsed_name in test_name:
                        status = parsed_stat
                        break

            test_passed = status == "PASSED"
            if test_passed:
                passed += 1
            details.append(
                {
                    "test": test_name,
                    "passed": test_passed,
                    "status": status or "NOT_FOUND",
                }
            )

        return TestResults(passed=passed, total=len(expected), details=details)

    ftp_results = _check_tests(fail_to_pass)
    ptp_results = _check_tests(pass_to_pass)

    return ftp_results, ptp_results


class SWEBenchProBenchmark:
    """SWE-bench Pro benchmark implementation.

    Multi-language benchmark for evaluating coding agents on real-world
    software engineering tasks across Python, Go, TypeScript, and JavaScript.
    """

    name = "swe-bench-pro"

    def __init__(
        self,
        dataset: str = "ScaleAI/SWE-bench_Pro",
        scripts_cache_dir: Path | None = None,
    ):
        """Initialize SWE-bench Pro benchmark.

        Args:
            dataset: HuggingFace dataset identifier.
            scripts_cache_dir: Override cache dir for run scripts repo.
        """
        self.dataset = dataset
        self._scripts_cache_dir = scripts_cache_dir
        self._scripts_repo_path: Path | None = None

    def _get_scripts_repo(self) -> Path:
        """Lazily clone and return the run scripts repo path."""
        if self._scripts_repo_path is None:
            self._scripts_repo_path = _ensure_run_scripts_repo(self._scripts_cache_dir)
        return self._scripts_repo_path

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

        Uses official run scripts from scaleapi/SWE-bench_Pro-os for all
        languages. Falls back to the standard evaluate_patch() for Python
        tasks when official scripts are not available.

        Args:
            env: Task environment.
            task: SWE-bench Pro task dictionary.
            solution: Unified diff patch to evaluate.

        Returns:
            Dictionary with evaluation results including 'resolved' boolean.
        """
        instance_id = task.get("instance_id", "")

        # Try to use official run scripts
        try:
            scripts_repo = self._get_scripts_repo()
            run_script, parser_script = _get_instance_scripts(scripts_repo, instance_id)
            return await self._evaluate_with_official_scripts(
                env, task, solution, run_script, parser_script
            )
        except FileNotFoundError:
            logger.info(
                "No official scripts for %s, falling back to standard evaluation",
                instance_id,
            )

        # Fallback for Python tasks without official scripts
        language = task.get("repo_language", "python").lower()
        if language == "python":
            eval_result: EvaluationResult = await evaluate_patch(env, task, solution)
            return self._eval_result_to_dict(eval_result)

        return {
            "resolved": False,
            "patch_applied": False,
            "eval_error": f"No official run scripts found for {instance_id}",
        }

    async def _evaluate_with_official_scripts(
        self,
        env: TaskEnvironment,
        task: dict[str, Any],
        patch: str,
        run_script: str,
        parser_script: str,
    ) -> dict[str, Any]:
        """Evaluate using official SWE-bench Pro run scripts.

        Args:
            env: Task environment.
            task: SWE-bench Pro task dictionary.
            patch: Unified diff patch to evaluate.
            run_script: Content of run_script.sh.
            parser_script: Content of parser.py.

        Returns:
            Dictionary with evaluation results.
        """
        language = task.get("repo_language", "python").lower()
        eval_workdir = "/app" if env.uses_prebuilt else None

        applied, error = await apply_patch(env, patch, workdir=eval_workdir)
        if not applied:
            return {"resolved": False, "patch_applied": False, "eval_error": error}

        test_patch = task.get("test_patch", "")
        if test_patch:
            await _apply_test_patch(env, test_patch, workdir=eval_workdir)

        # Run before_repo_set_cmd (last line only, matching official harness).
        # This typically restores specific test files from the fix commit,
        # e.g., "git checkout <commit> -- test/file.ts"
        await _run_before_repo_set_cmd(env, task, workdir=eval_workdir)

        # Reinstall package so patched code is active (SWE-bench Pro images
        # install into site-packages, not editable mode)
        if eval_workdir and language == "python":
            await env.exec_command(
                "pip install -e . -q 2>/dev/null || true",
                timeout=120,
                workdir=eval_workdir,
            )

        # Run tests using official scripts
        parsed_results = await _run_official_tests(
            env, task, run_script, parser_script, timeout=300
        )

        # Match against expected test lists
        fail_to_pass_str = get_test_list_field(task, "fail_to_pass")
        pass_to_pass_str = get_test_list_field(task, "pass_to_pass")
        fail_to_pass_tests = parse_test_list(fail_to_pass_str)
        pass_to_pass_tests = parse_test_list(pass_to_pass_str)

        ftp_results, ptp_results = _match_test_results(
            parsed_results, fail_to_pass_tests, pass_to_pass_tests
        )

        resolved = (
            ftp_results.passed == ftp_results.total
            and ftp_results.total > 0
            and ptp_results.passed == ptp_results.total
        )

        result: dict[str, Any] = {"resolved": resolved, "patch_applied": True}
        if ftp_results:
            result["fail_to_pass"] = {
                "passed": ftp_results.passed,
                "total": ftp_results.total,
            }
        if ptp_results:
            result["pass_to_pass"] = {
                "passed": ptp_results.passed,
                "total": ptp_results.total,
            }
        return result

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
