"""Main evaluation harness orchestrating parallel task execution."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from datasets import load_dataset
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from .config import HarnessConfig
from .docker_env import DockerEnvironmentManager, TaskEnvironment
from .evaluation import EvaluationResult, evaluate_patch
from .harnesses import AgentHarness, AgentResult, create_harness
from .log_formatter import InstanceLogWriter

console = Console()


@dataclass
class TaskResult:
    """Result for a single task."""

    instance_id: str
    mcp: dict[str, Any] | None = None
    baseline: dict[str, Any] | None = None


@dataclass
class EvaluationResults:
    """Complete evaluation results."""

    metadata: dict[str, Any]
    summary: dict[str, Any]
    tasks: list[TaskResult]


def agent_result_to_dict(
    result: AgentResult, eval_result: EvaluationResult | None
) -> dict[str, Any]:
    """Convert agent and evaluation results to dictionary."""
    data: dict[str, Any] = {
        "patch_generated": bool(result.patch),
        "tokens": {
            "input": result.tokens_input,
            "output": result.tokens_output,
        },
        "iterations": result.iterations,
        "tool_calls": result.tool_calls,
    }

    if result.tool_usage:
        data["tool_usage"] = result.tool_usage

    if result.error:
        data["error"] = result.error

    if result.messages:
        data["messages"] = result.messages

    if eval_result:
        data["resolved"] = eval_result.resolved
        data["patch_applied"] = eval_result.patch_applied

        if eval_result.fail_to_pass:
            data["fail_to_pass"] = {
                "passed": eval_result.fail_to_pass.passed,
                "total": eval_result.fail_to_pass.total,
            }

        if eval_result.pass_to_pass:
            data["pass_to_pass"] = {
                "passed": eval_result.pass_to_pass.passed,
                "total": eval_result.pass_to_pass.total,
            }

        if eval_result.error:
            data["eval_error"] = eval_result.error
    else:
        data["resolved"] = False
        data["patch_applied"] = False

    return data


def _create_mcp_agent(
    config: HarnessConfig,
    verbosity: int = 1,
    log_file: TextIO | InstanceLogWriter | None = None,
) -> AgentHarness:
    """Create the agent harness based on config.

    Args:
        config: Harness configuration.
        verbosity: Verbosity level (0=silent, 1=summary, 2=detailed).
        log_file: Optional file handle for writing raw JSON logs.

    Returns:
        Configured AgentHarness.
    """
    return create_harness(
        config.agent_harness,
        model=config.model,
        mcp_server=config.mcp_server,
        prompt=config.agent_prompt,
        max_iterations=config.max_iterations,
        verbosity=verbosity,
        log_file=log_file,
    )


def _create_baseline_agent(
    config: HarnessConfig,
    verbosity: int = 1,
    log_file: TextIO | InstanceLogWriter | None = None,
) -> AgentHarness:
    """Create a baseline agent (same as MCP agent, but without MCP server).

    Args:
        config: Harness configuration.
        verbosity: Verbosity level (0=silent, 1=summary, 2=detailed).
        log_file: Optional file handle for writing raw JSON logs.

    Returns:
        Configured AgentHarness without MCP server.
    """
    return create_harness(
        config.agent_harness,
        model=config.model,
        mcp_server=None,
        prompt=config.agent_prompt,
        max_iterations=config.max_iterations,
        verbosity=verbosity,
        log_file=log_file,
    )


async def run_single_task(
    task: dict[str, Any],
    config: HarnessConfig,
    docker_manager: DockerEnvironmentManager,
    run_mcp: bool,
    run_baseline: bool,
    verbose: bool,
    verbosity: int = 1,
    log_file: TextIO | None = None,
    log_dir: Path | None = None,
) -> TaskResult:
    """Run evaluation for a single task.

    Args:
        task: SWE-bench task dictionary.
        config: Harness configuration.
        docker_manager: Docker environment manager.
        run_mcp: Whether to run MCP evaluation.
        run_baseline: Whether to run baseline evaluation.
        verbose: Enable verbose output.
        verbosity: Verbosity level (0=silent, 1=summary, 2=detailed).
        log_file: Optional file handle for writing raw JSON logs.
        log_dir: Optional directory for per-instance JSON log files.

    Returns:
        TaskResult with results for both runs.
    """
    instance_id = task["instance_id"]
    result = TaskResult(instance_id=instance_id)

    if run_mcp:
        mcp_log_writer: InstanceLogWriter | None = None
        if log_dir:
            mcp_log_writer = InstanceLogWriter(log_dir, instance_id, "mcp")
        try:
            result.mcp = await _run_mcp_evaluation(
                task,
                config,
                docker_manager,
                verbose,
                verbosity,
                mcp_log_writer if mcp_log_writer else log_file,
            )
        finally:
            if mcp_log_writer:
                mcp_log_writer.close()

    if run_baseline:
        baseline_log_writer: InstanceLogWriter | None = None
        if log_dir:
            baseline_log_writer = InstanceLogWriter(log_dir, instance_id, "baseline")
        try:
            result.baseline = await _run_baseline_evaluation(
                task,
                config,
                docker_manager,
                verbose,
                verbosity,
                baseline_log_writer if baseline_log_writer else log_file,
            )
        finally:
            if baseline_log_writer:
                baseline_log_writer.close()

    return result


async def _run_mcp_evaluation(
    task: dict[str, Any],
    config: HarnessConfig,
    docker_manager: DockerEnvironmentManager,
    verbose: bool,
    verbosity: int = 1,
    log_file: TextIO | InstanceLogWriter | None = None,
) -> dict[str, Any]:
    """Run MCP agent evaluation."""
    env: TaskEnvironment | None = None
    try:
        env = await docker_manager.create_environment(task)

        agent = _create_mcp_agent(config, verbosity, log_file)

        instance_id = task.get("instance_id", "unknown")
        agent_result = await asyncio.wait_for(
            agent.solve(
                task,
                env.host_workdir,
                timeout=config.timeout_seconds,
                verbose=verbose,
                task_id=f"{instance_id}:mcp",
                env=env,  # Pass Docker environment for in-container execution
            ),
            timeout=config.timeout_seconds + 60,
        )

        if agent_result.patch:
            eval_result = await evaluate_patch(env, task, agent_result.patch)
        else:
            eval_result = None

        return agent_result_to_dict(agent_result, eval_result)

    except asyncio.TimeoutError:
        return {
            "resolved": False,
            "patch_applied": False,
            "error": "Timeout",
            "tokens": {"input": 0, "output": 0},
            "iterations": 0,
            "tool_calls": 0,
        }
    except Exception as e:
        return {
            "resolved": False,
            "patch_applied": False,
            "error": str(e),
            "tokens": {"input": 0, "output": 0},
            "iterations": 0,
            "tool_calls": 0,
        }
    finally:
        if env:
            await env.cleanup()


async def _run_baseline_evaluation(
    task: dict[str, Any],
    config: HarnessConfig,
    docker_manager: DockerEnvironmentManager,
    verbose: bool,
    verbosity: int = 1,
    log_file: TextIO | InstanceLogWriter | None = None,
) -> dict[str, Any]:
    """Run baseline agent evaluation (same harness as MCP, but without MCP server)."""
    env: TaskEnvironment | None = None
    try:
        env = await docker_manager.create_environment(task)

        agent = _create_baseline_agent(config, verbosity, log_file)

        instance_id = task.get("instance_id", "unknown")
        agent_result = await asyncio.wait_for(
            agent.solve(
                task,
                env.host_workdir,
                timeout=config.timeout_seconds,
                verbose=verbose,
                task_id=f"{instance_id}:baseline",
                env=env,  # Pass Docker environment for in-container execution
            ),
            timeout=config.timeout_seconds + 60,
        )

        if agent_result.patch:
            eval_result = await evaluate_patch(env, task, agent_result.patch)
        else:
            eval_result = None

        return agent_result_to_dict(agent_result, eval_result)

    except asyncio.TimeoutError:
        return {
            "resolved": False,
            "patch_applied": False,
            "error": "Timeout",
            "tokens": {"input": 0, "output": 0},
            "iterations": 0,
            "tool_calls": 0,
        }
    except Exception as e:
        return {
            "resolved": False,
            "patch_applied": False,
            "error": str(e),
            "tokens": {"input": 0, "output": 0},
            "iterations": 0,
            "tool_calls": 0,
        }
    finally:
        if env:
            await env.cleanup()


async def run_evaluation(
    config: HarnessConfig,
    run_mcp: bool = True,
    run_baseline: bool = True,
    verbose: bool = False,
    verbosity: int = 1,
    log_file: TextIO | None = None,
    log_dir: Path | None = None,
    task_ids: list[str] | None = None,
) -> EvaluationResults:
    """Run the full evaluation.

    Args:
        config: Harness configuration.
        run_mcp: Whether to run MCP evaluation.
        run_baseline: Whether to run baseline evaluation.
        verbose: Enable verbose output.
        verbosity: Verbosity level (0=silent, 1=summary, 2=detailed).
        log_file: Optional file handle for writing raw JSON logs.
        log_dir: Optional directory for per-instance JSON log files.
        task_ids: Specific task IDs to run (None for all).

    Returns:
        EvaluationResults with all results.
    """
    console.print(f"[dim]Loading dataset: {config.dataset}[/dim]")
    dataset = load_dataset(config.dataset, split="test")

    if task_ids:
        tasks = []
        for item in dataset:
            if item["instance_id"] in task_ids:
                tasks.append(item)
    else:
        tasks = list(dataset)

    if config.sample_size and len(tasks) > config.sample_size:
        tasks = tasks[: config.sample_size]

    console.print(f"[dim]Evaluating {len(tasks)} tasks[/dim]")
    console.print(f"[dim]Provider: {config.provider}, Harness: {config.agent_harness}[/dim]")

    docker_manager = DockerEnvironmentManager(use_prebuilt=config.use_prebuilt_images)

    results: list[TaskResult] = []
    semaphore = asyncio.Semaphore(config.max_concurrent)

    async def run_with_semaphore(task: dict[str, Any]) -> TaskResult:
        async with semaphore:
            return await run_single_task(
                task,
                config,
                docker_manager,
                run_mcp,
                run_baseline,
                verbose,
                verbosity,
                log_file,
                log_dir,
            )

    progress_console = Console(force_terminal=True)

    async_tasks = []
    for task in tasks:
        async_tasks.append(run_with_semaphore(task))

    try:
        if verbose:
            for coro in asyncio.as_completed(async_tasks):
                result = await coro
                results.append(result)
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=progress_console,
            ) as progress:
                main_task = progress.add_task("Evaluating tasks...", total=len(tasks))

                for coro in asyncio.as_completed(async_tasks):
                    result = await coro
                    results.append(result)
                    progress.update(main_task, advance=1)
    finally:
        await docker_manager.cleanup_all()

    mcp_resolved = 0
    baseline_resolved = 0
    mcp_total = 0
    baseline_total = 0

    for r in results:
        if r.mcp:
            mcp_total += 1
            if r.mcp.get("resolved"):
                mcp_resolved += 1
        if r.baseline:
            baseline_total += 1
            if r.baseline.get("resolved"):
                baseline_resolved += 1

    mcp_rate = mcp_resolved / mcp_total if mcp_total > 0 else 0
    baseline_rate = baseline_resolved / baseline_total if baseline_total > 0 else 0

    if baseline_rate > 0:
        improvement = ((mcp_rate - baseline_rate) / baseline_rate) * 100
        improvement_str = f"{improvement:+.1f}%"
    else:
        improvement_str = "N/A"

    return EvaluationResults(
        metadata={
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "model": config.model,
                "provider": config.provider,
                "agent_harness": config.agent_harness,
                "dataset": config.dataset,
                "sample_size": config.sample_size,
                "timeout_seconds": config.timeout_seconds,
                "max_iterations": config.max_iterations,
            },
            "mcp_server": {
                "command": config.mcp_server.command,
                "args": config.mcp_server.args,
                "args_note": "{workdir} is replaced with task repository path at runtime",
            },
        },
        summary={
            "mcp": {
                "resolved": mcp_resolved,
                "total": mcp_total,
                "rate": mcp_rate,
            },
            "baseline": {
                "resolved": baseline_resolved,
                "total": baseline_total,
                "rate": baseline_rate,
            },
            "improvement": improvement_str,
        },
        tasks=results,
    )
