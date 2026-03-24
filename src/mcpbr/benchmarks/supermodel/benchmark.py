"""SupermodelBenchmark -- PR-based analysis benchmark for mcpbr.

Supports multiple analysis types (dead-code, impact, test-coverage, circular-deps)
via endpoint plugins. Uses GitHub PRs for ground truth extraction and the Supermodel
API for pre-computed analysis in the enhanced (MCP) condition.
"""

import atexit
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from ...docker_env import DockerEnvironmentManager, TaskEnvironment
from .._bench_utils import extract_findings_from_text, init_git_workdir
from ..base import BenchmarkTask
from .api_client import call_supermodel_api
from .endpoints import get_endpoint
from .evaluation import compute_prf1
from .git_utils import clone_repo_at_commit, get_pre_merge_commit, zip_repo

logger = logging.getLogger("mcpbr.supermodel")

DEFAULT_GT_DIR = Path.home() / ".cache" / "mcpbr" / "supermodel_ground_truth"


class SupermodelBenchmark:
    """Supermodel analysis benchmark with PR-based ground truth.

    Implements the mcpbr Benchmark protocol. Each task is a GitHub PR
    where the ground truth is extracted from the diff.
    """

    name = "supermodel"
    evaluate_without_patch = True  # Uses REPORT.json, not git diff

    def __init__(
        self,
        analysis_type: str = "dead-code",
        tasks: list[dict[str, Any]] | None = None,
        supermodel_api_base: str = "https://api.supermodel.dev",
        supermodel_api_key: str | None = None,
        resolved_threshold: float = 0.8,
        ground_truth_dir: str | Path | None = None,
        supermodel_api_timeout: int = 900,
        **kwargs: Any,
    ):
        """Initialize the Supermodel benchmark.

        Args:
            analysis_type: Analysis endpoint to use (dead-code, impact, test-coverage,
                          circular-deps).
            tasks: List of task config dicts from YAML.
            supermodel_api_base: Base URL for Supermodel API.
            supermodel_api_key: API key (or set SUPERMODEL_API_KEY env var).
            resolved_threshold: Recall threshold to consider a task 'resolved' (precision is
                               reported but not required — the API returns many valid dead-code
                               candidates beyond the GT set, so precision is not a fair gate).
            ground_truth_dir: Directory to cache ground truth JSON files.
            supermodel_api_timeout: Max seconds to wait for Supermodel API (default 900).
            **kwargs: Additional keyword arguments (ignored for forward compat).
        """
        self.analysis_type = analysis_type
        self._tasks_config = tasks or []
        self.api_base = supermodel_api_base
        self.api_key = supermodel_api_key or os.environ.get("SUPERMODEL_API_KEY")
        self.api_timeout = supermodel_api_timeout
        self.resolved_threshold = resolved_threshold
        self.gt_dir = Path(ground_truth_dir) if ground_truth_dir else DEFAULT_GT_DIR
        self.gt_dir.mkdir(parents=True, exist_ok=True)

        self._endpoint = get_endpoint(analysis_type)
        self._loaded_tasks: list[dict[str, Any]] | None = None
        self._work_dir = Path(tempfile.mkdtemp(prefix="mcpbr_supermodel_"))
        atexit.register(self._cleanup_work_dir)

    def load_tasks(
        self,
        sample_size: int | None = None,
        task_ids: list[str] | None = None,
        _level: int | None = None,
        filter_difficulty: list[str] | None = None,
        filter_category: list[str] | None = None,
        filter_tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Load tasks from config and extract ground truth from PR diffs.

        Ground truth is cached in gt_dir to avoid repeated GitHub API calls.
        """
        _ = _level, filter_tags

        tasks = []
        for task_cfg in self._tasks_config:
            task_id = task_cfg["id"]
            repo = task_cfg.get("repo", "")
            language = task_cfg.get("language", "typescript")
            scope_prefix = task_cfg.get("scope_prefix")
            description = task_cfg.get("description", "")

            # Corpus mode: ground_truth_file points to a pre-existing GT JSON
            gt_file = task_cfg.get("ground_truth_file")
            if gt_file:
                gt_path = Path(gt_file).expanduser()
                if gt_path.exists():
                    with open(gt_path) as f:
                        gt = json.load(f)
                    logger.info(f"Loaded corpus GT: {len(gt)} items from {gt_path}")
                else:
                    logger.warning(f"GT file not found: {gt_path}, skipping {task_id}")
                    continue
            else:
                # PR mode: extract from diff
                pr_number = task_cfg["pr_number"]
                gt = self._load_ground_truth(task_id, repo, pr_number, language, scope_prefix)

            if not gt:
                logger.warning(f"No ground truth for {task_id}, skipping")
                continue

            task = {
                "instance_id": task_id,
                "repo": repo,
                "pr_number": task_cfg.get("pr_number"),
                "merge_commit": task_cfg.get("merge_commit", task_cfg.get("commit", "HEAD")),
                "commit": task_cfg.get("commit"),
                "clone_url": task_cfg.get("clone_url"),
                "language": language,
                "scope_prefix": scope_prefix,
                "description": description,
                "ground_truth": gt,
                "problem_statement": self._generate_baseline_problem_statement(task_cfg),
                "problem_statement_enhanced": self._generate_enhanced_problem_statement(task_cfg),
                "problem_statement_baseline": self._generate_baseline_problem_statement(task_cfg),
                "zip_exclude": task_cfg.get("zip_exclude", []),
                "cached_analysis": task_cfg.get("cached_analysis"),
            }
            tasks.append(task)

        if task_ids:
            task_id_set = set(task_ids)
            tasks = [t for t in tasks if t["instance_id"] in task_id_set]

        if filter_difficulty:
            difficulty_set = set(filter_difficulty)
            tasks = [t for t in tasks if t.get("difficulty", "hard") in difficulty_set]

        if filter_category:
            category_set = set(filter_category)
            tasks = [t for t in tasks if t.get("language", "typescript") in category_set]

        if sample_size and len(tasks) > sample_size:
            tasks = tasks[:sample_size]

        self._loaded_tasks = tasks
        return tasks

    def _load_ground_truth(
        self,
        task_id: str,
        repo: str,
        pr_number: int,
        language: str,
        scope_prefix: str | None,
    ) -> list[dict]:
        """Load cached ground truth or extract from PR diff."""
        ep_name = self._endpoint.name
        gt_path = self.gt_dir / f"{ep_name}_{task_id}.json"

        if gt_path.exists():
            with open(gt_path) as f:
                gt = json.load(f)
            logger.info(f"Loaded cached GT: {len(gt)} items from {gt_path}")
            return list(gt)

        logger.info(f"Extracting ground truth for {task_id} from PR diff...")
        gt = self._endpoint.extract_ground_truth(repo, pr_number, language, scope_prefix)

        with open(gt_path, "w") as f:
            json.dump(gt, f, indent=2)
        logger.info(f"Extracted {len(gt)} ground truth items -> {gt_path}")

        return list(gt)

    def _cleanup_work_dir(self) -> None:
        """Remove the temporary work directory."""
        if self._work_dir.exists():
            shutil.rmtree(self._work_dir, ignore_errors=True)

    def _generate_enhanced_problem_statement(self, task_cfg: dict) -> str:
        """Generate problem statement for the enhanced (graph-assisted) condition.

        Uses the v2 prompt if the endpoint provides one, otherwise falls back to
        the original v1 prompt from the endpoint plugin.
        """
        prompt = getattr(self._endpoint, "enhanced_prompt_v2", None)
        if prompt:
            return str(prompt)

        # Fallback: original v1 prompt from the endpoint plugin
        return str(self._endpoint.enhanced_prompt)

    def _generate_baseline_problem_statement(self, task_cfg: dict) -> str:
        """Generate problem statement for the baseline (manual analysis) condition.

        The agent must find dead code by reading and searching the codebase directly.
        """
        language = task_cfg.get("language", "typescript")

        ext = ".ts" if language == "typescript" else ".py"
        if language == "python":
            lang_hints = """- Functions in __all__ that are never actually imported by other modules
- Cleanup/utility functions whose associated state is never populated"""
        else:
            lang_hints = """- Exported functions/classes that are never imported by any other module
- Middleware or handlers that are defined but never registered with the router
- Methods on classes where the class itself is never instantiated from live code"""

        return f"""You are a code analyst. Find dead code in this {language} codebase.

Dead code = exported functions, classes, methods, interfaces, and constants that
are defined but never used in any meaningful execution path.

{lang_hints}

== STRATEGY ==

STEP 1: Get an overview of the codebase structure.
  - List the top-level directories and key source files.
  - Identify the main source directories (exclude node_modules, dist, build, tests).

STEP 2: Scan source files for exported symbols.
  - Focus on non-test, non-generated source files.
  - For each file, note exported functions, classes, interfaces, constants.

STEP 3: For each exported symbol, grep the codebase for references.
  - If it only appears in its own definition file (and possibly tests or
    barrel/index re-exports), it is likely dead.
  - Barrel re-exports (index.ts) do NOT count as real usage.
  - Type-only imports do NOT count as real usage.

STEP 4: Write REPORT.json EARLY (after scanning even a few files).
  - Write what you have so far, then continue scanning and UPDATE the file.
  - This ensures you always produce output even if you run out of iterations.

REPORT.json format:
{{
  "dead_code": [
    {{"file": "path/to/file{ext}", "name": "unusedFunc", "type": "function", "reason": "no callers found"}},
    ...
  ],
  "analysis_complete": true
}}

CRITICAL RULES:
- Type should be one of: function, class, method, const, interface, variable.
- When in doubt about whether something is dead, INCLUDE it.
- False positives are acceptable. Missing real dead code is NOT acceptable.
- Write REPORT.json after analyzing each batch — do NOT wait until the end.
- Prioritize breadth over depth: scan ALL source files before deep-diving any one."""

    def normalize_task(self, task: dict[str, Any]) -> BenchmarkTask:
        instance_id = task.get("instance_id", "unknown")
        return BenchmarkTask(
            task_id=instance_id,
            problem_statement=task.get("problem_statement", ""),
            repo=task.get("repo", "unknown"),
            commit=task.get("merge_commit", "HEAD"),
            metadata={
                "language": task.get("language", "typescript"),
                "analysis_type": self.analysis_type,
                "ground_truth_count": len(task.get("ground_truth", [])),
            },
        )

    async def create_environment(
        self,
        task: dict[str, Any],
        docker_manager: DockerEnvironmentManager,
    ) -> TaskEnvironment:
        """Create an isolated environment for the task.

        Clones repo at pre-merge commit, calls Supermodel API (or uses cached
        analysis), places analysis JSON in workdir, writes REPORT.json placeholder.
        """
        # Copy to avoid mutating the shared task dict (breaks A/B comparisons)
        task = {**task}
        task["problem_statement"] = task.get(
            "problem_statement_enhanced", task["problem_statement"]
        )

        instance_id = task["instance_id"]
        repo = task.get("repo", "")
        scope_prefix = task.get("scope_prefix")

        # Clone repo - corpus mode (clone_url + commit) or PR mode (repo + merge_commit)
        repo_dir = self._work_dir / f"repo-{instance_id}"
        if not repo_dir.exists():
            clone_url = task.get("clone_url")
            if clone_url:
                # Corpus mode: clone directly at specified commit
                commit = task.get("commit", "HEAD")
                logger.info(f"Corpus mode: cloning {clone_url} at {commit[:8]}")
                await clone_repo_at_commit(clone_url, commit, str(repo_dir))
            else:
                # PR mode: get pre-merge commit from merge commit
                merge_commit = task["merge_commit"]
                pre_merge = await get_pre_merge_commit(repo, merge_commit)
                logger.info(f"Pre-merge commit for {instance_id}: {pre_merge[:8]}")
                await clone_repo_at_commit(repo, pre_merge, str(repo_dir))

        # Create Docker environment
        await docker_manager._ensure_fallback_image()
        image_name = docker_manager.FALLBACK_IMAGE

        temp_dir = tempfile.TemporaryDirectory(prefix=f"mcpbr_{instance_id}_")
        docker_manager._temp_dirs.append(temp_dir)
        host_workdir = temp_dir.name

        # Copy repo to workdir (scoped if needed)
        # ignore_dangling_symlinks: skip broken symlinks (e.g. Cal.com .env)
        is_corpus = task.get("clone_url") is not None
        if scope_prefix:
            src_path = repo_dir / scope_prefix
            if src_path.is_dir():
                if is_corpus:
                    # Corpus mode: scoped content goes to workdir root so GT paths match
                    shutil.copytree(
                        str(src_path),
                        host_workdir,
                        dirs_exist_ok=True,
                        ignore_dangling_symlinks=True,
                    )
                else:
                    # PR mode: preserve directory structure for PR-relative paths
                    dest_path = Path(host_workdir) / scope_prefix
                    shutil.copytree(
                        str(src_path),
                        str(dest_path),
                        ignore_dangling_symlinks=True,
                    )
            else:
                shutil.copytree(
                    str(repo_dir),
                    host_workdir,
                    dirs_exist_ok=True,
                    ignore_dangling_symlinks=True,
                )
        else:
            shutil.copytree(
                str(repo_dir),
                host_workdir,
                dirs_exist_ok=True,
                ignore_dangling_symlinks=True,
            )

        # Write REPORT.json placeholder (key varies by analysis type)
        report_path = Path(host_workdir) / "REPORT.json"
        report_path.write_text(self._endpoint.report_placeholder())

        # Place analysis JSON in workdir for the agent
        # Priority: 1) cached_analysis file from task config, 2) Supermodel API call
        try:
            cached_path = task.get("cached_analysis")
            if cached_path and Path(cached_path).exists():
                with open(cached_path) as f:
                    analysis_json = json.load(f)
                print(
                    f"  Using cached analysis: {cached_path}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                exclude_patterns = task.get("zip_exclude", [])
                analysis_json = await self._get_analysis(
                    repo_dir,
                    instance_id,
                    scope_prefix,
                    exclude_patterns,
                    strip_prefix=is_corpus,
                )

            # --- Build analysis package for agent consumption ---
            # Keep reason + confidence so the agent can filter intelligently.
            # Also preserve metadata summary and entry points.
            keep_fields = {"file", "name", "type", "reason", "confidence"}

            # Find the candidate key
            candidate_key = None
            for k in ("deadCodeCandidates", "candidates", "items"):
                if k in analysis_json:
                    candidate_key = k
                    break

            all_candidates = analysis_json.get(candidate_key, []) if candidate_key else []

            # Extract metadata and entry points early (needed for filtering)
            metadata = analysis_json.get("metadata", {})

            # Pre-filter type/interface candidates (high FP rate from structural typing)
            type_interface_reasons = (
                "Type/interface with no references",
                "Type with no references",
                "Interface with no references",
            )
            before_count = len(all_candidates)
            all_candidates = [
                c
                for c in all_candidates
                if not any(str(c.get("reason", "")).startswith(r) for r in type_interface_reasons)
            ]
            type_filtered = before_count - len(all_candidates)
            if type_filtered:
                logger.info(
                    f"Pre-filtered {type_filtered} type/interface candidates for {instance_id}"
                )

            # NOTE: Cannot filter by reason — "Exported but file never imported"
            # contains both true positives AND false positives when
            # rootFilesCount is high. 16/20 GT items in tyr have this reason.
            # The signal is polluted at the parser level (import resolution
            # failure tags real dead code with the same reason as framework-
            # wired code). See issue #676 for details.
            root_files = metadata.get("rootFilesCount", 0) or 0

            # Build entry point set for cross-reference filtering
            ep_set = set()
            for ep in analysis_json.get("entryPoints", []):
                ep_file = ep.get("file", "")
                ep_name = ep.get("name", "")
                if ep_file and ep_name:
                    ep_set.add((ep_file, ep_name))

            # Drop candidates that match entry points
            if ep_set:
                before_ep = len(all_candidates)
                all_candidates = [
                    c
                    for c in all_candidates
                    if (c.get("file", ""), c.get("name", "")) not in ep_set
                ]
                ep_filtered = before_ep - len(all_candidates)
                if ep_filtered:
                    logger.info(f"Filtered {ep_filtered} entry point matches for {instance_id}")
            else:
                ep_filtered = 0

            # Slim candidates to keep_fields
            slimmed = [{k: v for k, v in c.items() if k in keep_fields} for c in all_candidates]

            # Build metadata summary for the agent
            reason_counts = Counter(c.get("reason", "") for c in all_candidates)
            confidence_counts = Counter(c.get("confidence", "") for c in all_candidates)
            entry_points = analysis_json.get("entryPoints", [])

            metadata_summary = {
                "totalCandidates": before_count,
                "includedCandidates": len(slimmed),
                "prefilteredTypeInterfaces": type_filtered,
                "entryPointFiltered": ep_filtered,
                "rootFilesCount": root_files,
                "reasonBreakdown": dict(reason_counts.most_common()),
                "confidenceBreakdown": dict(confidence_counts.most_common()),
            }

            # Slim entry points for the whitelist
            ep_keep = {"file", "name", "type", "reason"}
            slim_entry_points = [
                {k: v for k, v in ep.items() if k in ep_keep} for ep in entry_points[:200]
            ]

            # Write all candidates directly into a single analysis file.
            analysis_data = {
                "metadataSummary": metadata_summary,
                "deadCodeCandidates": slimmed,
                "entryPoints": slim_entry_points,
            }
            index_path = Path(host_workdir) / self._endpoint.analysis_filename
            index_path.write_text(json.dumps(analysis_data, indent=2))

            logger.info(
                f"Placed analysis for {instance_id}: {len(slimmed)} candidates, "
                f"{len(slim_entry_points)} entry points "
                f"(filtered: {type_filtered} types, {ep_filtered} entry points)"
            )
        except Exception as e:
            logger.error(f"Failed to get Supermodel analysis for {instance_id}: {e}")
            print(
                f"\n*** SUPERMODEL ANALYSIS FAILED for {instance_id} ***\n{traceback.format_exc()}",
                file=sys.stderr,
                flush=True,
            )

        # Start Docker container
        container_name = f"mcpbr-{docker_manager._session_id}-{instance_id}"
        container_workdir = "/workspace"

        container = docker_manager.client.containers.run(
            image_name,
            command="tail -f /dev/null",
            name=container_name,
            detach=True,
            network_mode="bridge",
            volumes={host_workdir: {"bind": "/workspace", "mode": "rw"}},
            working_dir=container_workdir,
            remove=False,
            labels={
                "mcpbr": "true",
                "session_id": docker_manager._session_id,
                "instance_id": instance_id,
            },
        )

        docker_manager._containers.append(container)

        env = TaskEnvironment(
            container=container,
            workdir=container_workdir,
            host_workdir=host_workdir,
            instance_id=instance_id,
            uses_prebuilt=False,
            claude_cli_installed=False,
        )

        # Init git so the harness can track modifications
        init_git_workdir(host_workdir)

        return env

    async def _get_analysis(
        self,
        repo_dir: Path,
        task_id: str,
        scope_prefix: str | None,
        exclude_patterns: list[str] | None = None,
        strip_prefix: bool = True,
    ) -> dict:
        """Call Supermodel API and return parsed/filtered analysis.

        Results are cached in gt_dir/{task_id}_analysis.json keyed by zip hash
        so subsequent runs skip the API call.
        """
        zip_path = str(self._work_dir / f"{task_id}.zip")
        await zip_repo(str(repo_dir), zip_path, scope_prefix, exclude_patterns)

        # Check cache
        with open(zip_path, "rb") as f:
            zip_hash = hashlib.sha256(f.read()).hexdigest()[:12]
        cache_path = self.gt_dir / f"{task_id}_analysis_{zip_hash}.json"
        if cache_path.exists():
            logger.info(f"Using cached analysis: {cache_path}")
            with open(cache_path) as f:
                return dict(json.load(f))

        raw_response = await call_supermodel_api(
            endpoint_path=self._endpoint.api_path,
            zip_path=zip_path,
            api_base=self.api_base,
            api_key=self.api_key,
            max_poll_time=self.api_timeout,
        )

        result = self._endpoint.parse_api_response(raw_response)

        # Strip scope_prefix from file paths so they match the workdir layout.
        # Only in corpus mode (strip_prefix=True): workdir content is at root.
        # In PR mode (strip_prefix=False): scope_prefix dir is preserved in workdir.
        if scope_prefix and strip_prefix:
            prefix = scope_prefix.rstrip("/") + "/"
            for key in ("deadCodeCandidates", "candidates", "items"):
                if key in result:
                    for item in result[key]:
                        fp = item.get("file", "")
                        if fp.startswith(prefix):
                            item["file"] = fp[len(prefix) :]

        # Cache the result for future runs
        cache_path.write_text(json.dumps(result, indent=2))
        logger.info(f"Cached analysis at {cache_path}")

        return result

    async def evaluate(
        self,
        env: TaskEnvironment,
        task: dict[str, Any],
        solution: str,
    ) -> dict[str, Any]:
        """Evaluate by reading REPORT.json from the workspace and computing P/R/F1."""
        ground_truth = task.get("ground_truth", [])
        key_fields = self._endpoint.key_fields

        # Read REPORT.json from host
        report_path = Path(env.host_workdir) / "REPORT.json"
        agent_findings: list[dict[str, Any]] = []

        if report_path.exists():
            try:
                with open(report_path) as f:
                    report = json.load(f)
                agent_findings = report.get(self._endpoint.findings_key, [])
            except (json.JSONDecodeError, OSError):
                agent_findings = self._extract_findings_from_text(solution)
        else:
            agent_findings = self._extract_findings_from_text(solution)

        # Compute P/R/F1
        metrics = compute_prf1(agent_findings, ground_truth, key_fields)

        precision = metrics["precision"]
        recall = metrics["recall"]
        resolved = recall >= self.resolved_threshold

        # Log results
        print(f"\n{'=' * 50}")
        print(f"SUPERMODEL EVALUATION - {env.instance_id} ({self.analysis_type})")
        print(f"  Found: {metrics['found']} items")
        print(f"  Expected: {metrics['expected']} items")
        print(f"  True Positives: {metrics['true_positives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        print(f"  Precision: {precision * 100:.1f}%")
        print(f"  Recall: {recall * 100:.1f}%")
        print(f"  F1 Score: {metrics['f1_score'] * 100:.1f}%")
        print(f"  Resolved: {resolved}")
        print(f"{'=' * 50}\n")

        return {
            "resolved": resolved,
            **metrics,
        }

    def _extract_findings_from_text(self, text: str) -> list[dict[str, Any]]:
        """Extract findings from text/patch content as fallback."""
        return extract_findings_from_text(text, self._endpoint.findings_key)

    def get_prebuilt_image(self, task: dict[str, Any]) -> str | None:
        return None

    def get_prompt_template(self) -> str:
        return "{problem_statement}"
