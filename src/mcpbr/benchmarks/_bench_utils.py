"""Shared utilities for benchmark implementations."""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger("mcpbr.benchmarks")


def extract_findings_from_text(text: str, findings_key: str = "dead_code") -> list[dict[str, Any]]:
    """Extract findings array from text/patch content by locating a JSON key.

    Searches for a JSON key (e.g. "dead_code") and extracts the associated array
    using bracket-depth matching. Handles brackets inside JSON strings correctly.

    Args:
        text: Raw text that may contain a JSON object with the findings key.
        findings_key: The JSON key whose array value to extract.

    Returns:
        List of finding dicts, or empty list if not found/parseable.
    """
    findings: list[dict[str, Any]] = []
    try:
        marker = f'"{findings_key}"'
        start = text.find(marker)
        if start == -1:
            return findings
        arr_start = text.find("[", start)
        if arr_start == -1:
            return findings
        # Bracket-depth matching that respects JSON strings
        depth = 0
        in_string = False
        escape_next = False
        for i, c in enumerate(text[arr_start:], arr_start):
            if escape_next:
                escape_next = False
                continue
            if c == "\\":
                if in_string:
                    escape_next = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    arr_text = text[arr_start : i + 1]
                    parsed = json.loads(arr_text)
                    if isinstance(parsed, list):
                        findings = parsed
                    break
    except (json.JSONDecodeError, ValueError):
        pass
    return findings


def init_git_workdir(host_workdir: str, timeout: int = 30) -> None:
    """Initialize a git repo in a workdir so the harness can track modifications.

    Args:
        host_workdir: Path to the working directory.
        timeout: Timeout in seconds for each git command.
    """
    subprocess.run(
        ["git", "init"], cwd=host_workdir, capture_output=True, check=False, timeout=timeout
    )
    subprocess.run(
        ["git", "config", "user.email", "mcpbr@test.com"],
        cwd=host_workdir,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    subprocess.run(
        ["git", "config", "user.name", "MCPBR"],
        cwd=host_workdir,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    subprocess.run(
        ["git", "add", "-A"],
        cwd=host_workdir,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial"],
        cwd=host_workdir,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def safe_write_file(host_workdir: str, file_path: str, content: str) -> None:
    """Write a file within host_workdir, raising if the path escapes containment.

    Args:
        host_workdir: Root directory that all writes must stay within.
        file_path: Relative path of the file to write.
        content: File content.

    Raises:
        ValueError: If the resolved path is outside host_workdir.
    """
    root = Path(host_workdir).resolve()
    full_path = (root / file_path).resolve()
    if not full_path.is_relative_to(root):
        raise ValueError(f"Path traversal detected: {file_path!r} escapes {host_workdir!r}")
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(content)
