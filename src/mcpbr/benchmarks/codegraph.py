"""Code graph navigation benchmark.

Evaluates whether MCP tools help agents navigate code knowledge graphs.
Agents start at a node in a Supermodel IR code graph and must find a
target node using graph exploration tools. Tasks span 10 well-known
open-source repos across 3 difficulty tiers (easy, medium, hard).

Dataset: supermodeltools/codegraph-bench on HuggingFace

The benchmark pre-loads graph files into the Docker container in
Supermodel MCP server cache format. The MCP server reads from
SUPERMODEL_CACHE_DIR at startup with --no-api-fallback.
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from datasets import load_dataset

from ..docker_env import DockerEnvironmentManager, TaskEnvironment
from .base import BenchmarkTask

logger = logging.getLogger(__name__)


class CodeGraphBenchmark:
    """Code graph navigation benchmark.

    Tasks involve navigating a code knowledge graph from a start node to
    a target node. Agents use MCP tools to explore node properties,
    list neighbors, and traverse relationships. Evaluation checks whether
    the agent submitted the correct target node and measures efficiency
    (steps taken vs optimal path length).
    """

    name = "codegraph"

    CACHE_DIR = "/workspace/supermodel-cache"

    def __init__(
        self,
        dataset: str = "supermodeltools/codegraph-bench",
        subset: str = "default",
    ):
        """Initialize codegraph benchmark.

        Args:
            dataset: HuggingFace dataset identifier.
            subset: Dataset subset/config name.
        """
        self.dataset = dataset
        self.subset = subset
        self._graphs_cache: dict[str, str] | None = None

    def _load_graphs_from_hf(self) -> dict[str, str]:
        """Load and cache all graph data from HuggingFace."""
        if self._graphs_cache is not None:
            return self._graphs_cache

        self._graphs_cache = {}
        try:
            graphs_dataset = load_dataset(self.dataset, self.subset, split="graphs")
            for row in graphs_dataset:
                graph_file = row.get("graph_file", "")
                graph_json = row.get("graph_json", "")
                if graph_file and graph_json:
                    self._graphs_cache[graph_file] = graph_json
            logger.info("Loaded %d graphs from HuggingFace", len(self._graphs_cache))
        except Exception:
            logger.exception("Failed to load graphs from HuggingFace")

        return self._graphs_cache

    def load_tasks(
        self,
        sample_size: int | None = None,
        task_ids: list[str] | None = None,
        level: int | None = None,
        filter_difficulty: list[str] | None = None,
        filter_category: list[str] | None = None,
        filter_tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Load code graph navigation tasks from HuggingFace.

        Args:
            sample_size: Maximum number of tasks to load.
            task_ids: Specific task IDs to load.
            level: Unused.
            filter_difficulty: Filter by difficulty (easy, medium, hard).
            filter_category: Filter by repo name.
            filter_tags: Filter by language tags.

        Returns:
            List of task dictionaries.
        """
        dataset = load_dataset(self.dataset, self.subset, split="test")

        tasks = list(dataset)

        # Filter by specific task IDs
        if task_ids:
            task_id_set = set(task_ids)
            tasks = [t for t in tasks if t.get("task_id") in task_id_set]

        # Filter by difficulty
        if filter_difficulty:
            diff_set = set(filter_difficulty)
            tasks = [t for t in tasks if t.get("difficulty") in diff_set]

        # Filter by repo (via category filter)
        if filter_category:
            cat_set = set(filter_category)
            tasks = [t for t in tasks if t.get("repo") in cat_set]

        # Filter by language (via tags filter)
        if filter_tags:
            tag_set = set(filter_tags)
            tasks = [t for t in tasks if t.get("language") in tag_set]

        # Apply sample size
        if sample_size is not None and len(tasks) > sample_size:
            tasks = tasks[:sample_size]

        # Augment with fields the harness expects
        augmented = []
        for task in tasks:
            aug = dict(task)
            aug["instance_id"] = task["task_id"]
            aug["problem_statement"] = self._generate_problem_statement(aug)
            augmented.append(aug)

        return augmented

    def normalize_task(self, task: dict[str, Any]) -> BenchmarkTask:
        """Convert codegraph task to normalized format.

        Args:
            task: Codegraph task dictionary.

        Returns:
            Normalized BenchmarkTask.
        """
        instance_id = task.get("instance_id") or task.get("task_id")
        if not instance_id:
            msg = f"Task missing 'task_id' field: {task.keys()}"
            raise ValueError(msg)

        return BenchmarkTask(
            task_id=instance_id,
            problem_statement=self._generate_problem_statement(task),
            repo=task.get("repo", "codegraph/unknown"),
            commit="HEAD",
            metadata={
                "difficulty": task.get("difficulty"),
                "target_node": task.get("target_node"),
                "start_node": task.get("start_node"),
                "optimal_steps": task.get("optimal_steps"),
                "graph_file": task.get("graph_file"),
                "language": task.get("language"),
            },
        )

    def _generate_problem_statement(self, task: dict[str, Any]) -> str:
        """Generate the navigation problem statement for the agent.

        Args:
            task: Task dictionary.

        Returns:
            Problem statement describing the navigation task.
        """
        repo = task.get("repo", "unknown")
        description = task.get("description", "")
        start_node = task.get("start_node", "")
        difficulty = task.get("difficulty", "")

        statement = (
            f"You are navigating the code graph for the {repo} repository.\n\n"
            f"TASK: {description}\n\n"
            f"Your starting node ID is: {start_node}\n\n"
            f"Difficulty: {difficulty}\n\n"
            "Use the Supermodel MCP tools to explore the graph:\n"
            "- Use 'overview' to get the architectural map of the codebase\n"
            "- Use 'symbol_context' to inspect specific functions, classes, or modules\n"
            "- Navigate relationships: imports, calls, contains, belongsTo\n\n"
            "When you have found the target node, submit your answer by writing:\n"
            "SUBMIT: <node_id>\n\n"
            "where <node_id> is the exact ID of the target node."
        )

        return statement

    async def create_environment(
        self,
        task: dict[str, Any],
        docker_manager: DockerEnvironmentManager,
    ) -> TaskEnvironment:
        """Create environment with graph data pre-loaded in MCP cache format.

        Loads the graph from HuggingFace, converts to Supermodel MCP server
        cache format, and writes to SUPERMODEL_CACHE_DIR in the container.

        Args:
            task: Task dictionary.
            docker_manager: Docker environment manager.

        Returns:
            TaskEnvironment with graph cache ready for MCP server.
        """
        instance_id = task.get("instance_id", "codegraph_unknown")

        temp_task = {
            "instance_id": instance_id,
            "repo": task.get("repo", "codegraph/unknown"),
            "base_commit": "HEAD",
        }

        env = await docker_manager.create_environment(temp_task)

        # Pre-load graph data in MCP cache format
        await self._setup_environment(env, task)

        return env

    async def _setup_environment(self, env: TaskEnvironment, task: dict[str, Any]) -> None:
        """Pre-load graph into container in Supermodel MCP server cache format.

        The MCP server reads from SUPERMODEL_CACHE_DIR at startup.
        Cache format: {version, repoName, commitHash, savedAt, raw: <SupermodelIR>}

        Args:
            env: Task environment.
            task: Task dictionary with graph_file and repo.
        """
        graph_file = task.get("graph_file", "")
        repo = task.get("repo", "")
        if not graph_file or not repo:
            logger.warning("No graph_file/repo for task %s", task.get("task_id"))
            return

        # Create cache directory
        await env.exec_command(f"mkdir -p {self.CACHE_DIR}", timeout=10)

        # Load graph data from HuggingFace
        graphs = self._load_graphs_from_hf()
        raw_json = graphs.get(graph_file)
        if not raw_json:
            logger.warning("Graph %s not found in dataset", graph_file)
            return

        # Parse the wrapper to extract SupermodelIR
        try:
            wrapper = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
            data = wrapper.get("data", wrapper)
            result = data.get("result") or data
        except (json.JSONDecodeError, AttributeError):
            logger.exception("Failed to parse graph %s", graph_file)
            return

        # Convert to MCP server cache format
        cache_name = repo.replace("/", "__")
        cache_entry = {
            "version": 1,
            "repoName": cache_name,
            "commitHash": None,
            "savedAt": datetime.now(timezone.utc).isoformat(),
            "raw": result,
        }

        cache_json = json.dumps(cache_entry)
        cache_path = f"supermodel-cache/{cache_name}.json"
        await env.write_file(cache_path, cache_json)
        logger.info("Loaded graph %s into cache as %s.json", graph_file, cache_name)

    async def evaluate(
        self,
        env: TaskEnvironment,
        task: dict[str, Any],
        solution: str,
    ) -> dict[str, Any]:
        """Evaluate whether the agent found the correct target node.

        Extracts the submitted node ID from the agent's output and compares
        against the ground truth target. Also computes efficiency score.

        Args:
            env: Task environment.
            task: Task dictionary with target_node.
            solution: Agent's full output text.

        Returns:
            Dictionary with 'resolved', efficiency metrics, and details.
        """
        target_node = task.get("target_node", "")
        if not target_node:
            return {
                "resolved": False,
                "error": "No target_node in task definition",
            }

        # Extract submitted node from agent output
        submitted = self._extract_submission(solution)

        if submitted is None:
            return {
                "resolved": False,
                "error": "Could not extract SUBMIT: <node_id> from agent output",
                "agent_output_tail": solution[-500:] if solution else "",
            }

        correct = submitted == target_node
        optimal_steps = task.get("optimal_steps", 1)

        # Count steps from agent output (approximate by counting tool calls)
        steps = self._count_steps(solution)

        # Efficiency: optimal / actual (1.0 = perfect, <1 = took longer)
        efficiency = optimal_steps / steps if steps > 0 and correct else 0.0

        return {
            "resolved": correct,
            "submitted_node": submitted,
            "target_node": target_node,
            "steps": steps,
            "optimal_steps": optimal_steps,
            "efficiency": round(efficiency, 3),
            "difficulty": task.get("difficulty", ""),
            "repo": task.get("repo", ""),
        }

    def _extract_submission(self, text: str) -> str | None:
        """Extract the submitted node ID from agent output.

        Looks for patterns like:
        - SUBMIT: node_id
        - SUBMIT: "node_id"
        - submit: node_id

        Args:
            text: Agent's full output.

        Returns:
            Node ID string or None if not found.
        """
        if not text:
            return None

        # Pattern: SUBMIT: <node_id> (with optional quotes)
        match = re.search(
            r"SUBMIT:\s*[\"']?([a-zA-Z0-9_:/.@<>()[\]{},=+\-# ]+?)[\"']?\s*$",
            text,
            re.MULTILINE | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()

        # Fallback: last line containing "submit" with a node-like string
        for line in reversed(text.splitlines()):
            if "submit" in line.lower():
                # Try to extract anything after "submit"
                m = re.search(r"submit\S*\s+(.+)", line, re.IGNORECASE)
                if m:
                    return m.group(1).strip().strip("\"'")

        return None

    def _count_steps(self, text: str) -> int:
        """Estimate the number of navigation steps from agent output.

        Counts MCP tool calls as a proxy for steps taken.

        Args:
            text: Agent's full output.

        Returns:
            Estimated step count (minimum 1).
        """
        if not text:
            return 1

        # Count tool call patterns in the output
        tool_calls = len(re.findall(r"(?:tool_use|tool_call|<tool>|Tool:|Calling)", text, re.IGNORECASE))
        return max(tool_calls, 1)

    def get_prebuilt_image(self, task: dict[str, Any]) -> str | None:
        """No pre-built images for codegraph.

        Args:
            task: Task dictionary.

        Returns:
            None.
        """
        return None

    def get_prompt_template(self) -> str:
        """Get codegraph prompt template.

        Returns:
            Prompt template with navigation instructions.
        """
        return (
            "You are a code navigation agent. Your task is to navigate a code knowledge graph\n"
            "to find a specific target node.\n\n"
            "{problem_statement}\n\n"
            "INSTRUCTIONS:\n"
            "- Use the MCP tools to explore the graph (inspect nodes, list neighbors, traverse edges)\n"
            "- Reason about code structure: files contain functions, domains group related code,\n"
            "  imports/calls/contains relationships connect nodes\n"
            "- Navigate efficiently — fewer steps is better\n"
            "- When you find the target, write EXACTLY: SUBMIT: <node_id>\n"
            "- The node_id must be the exact ID from the graph, not a description\n"
        )

    def get_default_sandbox_level(self) -> str | None:
        """Codegraph tasks don't execute untrusted code.

        Returns:
            None (use global default).
        """
        return None
