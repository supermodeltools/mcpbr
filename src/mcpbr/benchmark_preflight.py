"""Preflight validation for benchmarks.

Validates that golden patches pass all tests in Docker environments before
running agent evaluations. This catches environment/configuration issues
early, ensuring evaluation infrastructure works correctly.
"""

import asyncio
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Any

from .docker_env import DockerEnvironmentManager, TaskEnvironment
from .evaluation import (
    TestResults,
    _apply_test_patch,
    apply_patch,
    get_test_list_field,
    parse_test_list,
    run_tests,
)

logger = logging.getLogger(__name__)


@dataclass
class PreflightResult:
    """Result of a single preflight instance check."""

    instance_id: str
    status: str  # "passed", "failed", "error"
    fail_to_pass_passed: int = 0
    fail_to_pass_total: int = 0
    pass_to_pass_passed: int = 0
    pass_to_pass_total: int = 0
    error: str | None = None
    language: str = "unknown"


@dataclass
class PreflightReport:
    """Aggregate preflight validation report."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    results: list[PreflightResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.total == 0:
            return 0.0
        return (self.passed / self.total) * 100.0


async def _run_preflight_tests(
    env: TaskEnvironment,
    tests: list[str],
    language: str,
    timeout: int = 300,
    uses_conda: bool = False,
    workdir: str | None = None,
    repo: str | None = None,
) -> TestResults:
    """Run tests using the appropriate language-specific runner.

    For Python, delegates to the standard run_tests(). For Go, JavaScript,
    and TypeScript, builds language-specific commands (go test, npx jest).

    Args:
        env: Task environment.
        tests: List of test identifiers.
        language: Programming language.
        timeout: Timeout per test in seconds.
        uses_conda: Whether to activate conda environment.
        workdir: Working directory inside container.
        repo: Repository name (used for Python test specs).

    Returns:
        TestResults with pass/fail counts.
    """
    if language == "python":
        return await run_tests(
            env,
            tests,
            timeout=timeout,
            uses_prebuilt=uses_conda,
            workdir=workdir,
            repo=repo,
        )

    # Non-Python: use language-specific test commands
    from .benchmarks.swebench_pro import _build_pro_test_command, _detect_js_runner

    if not tests:
        return TestResults(passed=0, total=0, details=[])

    # Detect JS/TS test runner once per instance
    js_runner = "jest"
    if language in ("typescript", "javascript", "ts", "js"):
        js_runner = await _detect_js_runner(env, workdir=workdir)

    results = []
    passed = 0

    for test in tests:
        test_cmd = _build_pro_test_command(test, language, uses_conda, js_runner=js_runner)
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


async def _prune_docker_images() -> None:
    """Remove unused Docker images to free disk space.

    Called after each preflight instance to prevent disk exhaustion.
    Each SWE-bench Pro image is ~1.5GB and each instance uses a unique image,
    so pruning after cleanup is critical for processing many instances.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "image",
            "prune",
            "-af",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        await proc.wait()
    except Exception:
        logger.debug("Failed to prune Docker images")


async def _check_single_instance(
    benchmark: Any,
    task: dict[str, Any],
    docker_manager: DockerEnvironmentManager,
    timeout: int = 300,
) -> PreflightResult:
    """Validate a single benchmark instance by applying the golden patch.

    Args:
        benchmark: Benchmark instance with create_environment method.
        task: Task dictionary with patch, test_patch, fail_to_pass, pass_to_pass.
        docker_manager: Docker environment manager.
        timeout: Timeout per test in seconds.

    Returns:
        PreflightResult for this instance.
    """
    instance_id = task.get("instance_id", "unknown")
    language = task.get("repo_language", "python").lower()
    env: TaskEnvironment | None = None

    try:
        # Create Docker environment (skip Claude CLI install — not needed for preflight)
        preflight_task = dict(task)
        preflight_task["_skip_cli_install"] = True
        env = await benchmark.create_environment(preflight_task, docker_manager)

        # Determine eval workdir: SWE-bench Pro images use /app (indicated by
        # dockerhub_tag), standard SWE-bench uses /testbed.
        eval_workdir: str | None
        if env.uses_prebuilt:
            if task.get("dockerhub_tag"):
                eval_workdir = "/app"
            else:
                eval_workdir = "/testbed"
        else:
            eval_workdir = None

        # Apply golden patch
        golden_patch = task.get("patch", "")
        if not golden_patch:
            return PreflightResult(
                instance_id=instance_id,
                status="error",
                error="No golden patch found in task",
                language=language,
            )

        applied, error = await apply_patch(env, golden_patch, workdir=eval_workdir)
        if not applied:
            return PreflightResult(
                instance_id=instance_id,
                status="failed",
                error=f"Golden patch failed to apply: {error}",
                language=language,
            )

        # Apply test patch
        test_patch = task.get("test_patch", "")
        if test_patch:
            await _apply_test_patch(env, test_patch, workdir=eval_workdir)

        # Reinstall package in editable mode so patched code is used.
        # SWE-bench Pro images install the package into site-packages;
        # without this step, tests would import the old (unpatched) code.
        if eval_workdir and language == "python":
            await env.exec_command(
                "pip install -e . -q 2>/dev/null || true",
                timeout=120,
                workdir=eval_workdir,
            )

        # Parse test lists (handle both uppercase and lowercase field names)
        fail_to_pass_str = get_test_list_field(task, "fail_to_pass")
        pass_to_pass_str = get_test_list_field(task, "pass_to_pass")
        fail_to_pass_tests = parse_test_list(fail_to_pass_str)
        pass_to_pass_tests = parse_test_list(pass_to_pass_str)

        # SWE-bench Pro images don't use conda, so skip conda activation
        # even though uses_prebuilt is True (it only means "image was pulled")
        uses_conda = env.uses_prebuilt and not task.get("dockerhub_tag")

        # Run fail_to_pass tests (all must PASS with golden patch)
        ftp_results = await _run_preflight_tests(
            env,
            fail_to_pass_tests,
            language=language,
            timeout=timeout,
            uses_conda=uses_conda,
            workdir=eval_workdir,
            repo=task.get("repo"),
        )

        # Run pass_to_pass tests (all must still PASS)
        ptp_results = await _run_preflight_tests(
            env,
            pass_to_pass_tests[:10],
            language=language,
            timeout=timeout,
            uses_conda=uses_conda,
            workdir=eval_workdir,
            repo=task.get("repo"),
        )

        # Determine status
        all_ftp_pass = ftp_results.passed == ftp_results.total and ftp_results.total > 0
        all_ptp_pass = ptp_results.passed == ptp_results.total

        if all_ftp_pass and all_ptp_pass:
            status = "passed"
            error_msg = None
        else:
            status = "failed"
            parts = []
            if not all_ftp_pass:
                parts.append(f"fail_to_pass: {ftp_results.passed}/{ftp_results.total} passed")
            if not all_ptp_pass:
                parts.append(f"pass_to_pass: {ptp_results.passed}/{ptp_results.total} passed")
            error_msg = "; ".join(parts)

        return PreflightResult(
            instance_id=instance_id,
            status=status,
            fail_to_pass_passed=ftp_results.passed,
            fail_to_pass_total=ftp_results.total,
            pass_to_pass_passed=ptp_results.passed,
            pass_to_pass_total=ptp_results.total,
            error=error_msg,
            language=language,
        )

    except Exception as e:
        logger.exception(f"Preflight error for {instance_id}")
        return PreflightResult(
            instance_id=instance_id,
            status="error",
            error=str(e),
            language=language,
        )

    finally:
        if env is not None:
            try:
                await env.cleanup()
            except Exception:
                logger.warning(f"Failed to clean up container for {instance_id}")
        # Prune unused images to free disk space (each image is ~1.5GB)
        await _prune_docker_images()


async def run_benchmark_preflight(
    benchmark: Any,
    tasks: list[dict[str, Any]],
    docker_manager: DockerEnvironmentManager,
    max_concurrent: int = 4,
    timeout: int = 300,
    fail_fast: bool = False,
) -> PreflightReport:
    """Run preflight validation on benchmark tasks.

    Applies golden patches and verifies all tests pass for each instance.

    Args:
        benchmark: Benchmark instance.
        tasks: List of task dictionaries to validate.
        docker_manager: Docker environment manager.
        max_concurrent: Maximum concurrent validations.
        timeout: Timeout per test in seconds.
        fail_fast: Stop on first failure.

    Returns:
        PreflightReport with aggregate results.
    """
    report = PreflightReport(total=len(tasks))
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _check_with_semaphore(task: dict[str, Any]) -> PreflightResult:
        async with semaphore:
            return await _check_single_instance(benchmark, task, docker_manager, timeout)

    if fail_fast:
        # Sequential execution with early exit
        for task in tasks:
            result = await _check_with_semaphore(task)
            report.results.append(result)
            if result.status == "passed":
                report.passed += 1
            elif result.status == "failed":
                report.failed += 1
                break
            else:
                report.errors += 1
                break
    else:
        # Concurrent execution
        coros = [_check_with_semaphore(task) for task in tasks]
        results = await asyncio.gather(*coros, return_exceptions=True)

        for r in results:
            if isinstance(r, BaseException):
                report.errors += 1
                report.results.append(
                    PreflightResult(
                        instance_id="unknown",
                        status="error",
                        error=str(r),
                    )
                )
            else:
                preflight_result: PreflightResult = r
                report.results.append(preflight_result)
                if preflight_result.status == "passed":
                    report.passed += 1
                elif preflight_result.status == "failed":
                    report.failed += 1
                else:
                    report.errors += 1

    return report
