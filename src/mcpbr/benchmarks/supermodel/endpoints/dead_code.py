"""Dead code detection endpoint plugin.

Ground truth: PRs that remove dead code (unused exported functions/classes/consts).
Tuple format: (file, symbol_name)
"""

import re
from dataclasses import dataclass

from .base import EndpointPlugin

# Patterns for TypeScript/JavaScript exported declarations
TS_PATTERNS = [
    (r"^-\s*export\s+(?:async\s+)?function\s+(\w+)", "function"),
    (r"^-\s*export\s+class\s+(\w+)", "class"),
    (r"^-\s*export\s+const\s+(\w+)\s*[=:]", "const"),
    (r"^-\s*export\s+default\s+(?:async\s+)?function\s+(\w+)", "function"),
    (r"^-\s*export\s+default\s+class\s+(\w+)", "class"),
    (r"^-\s*export\s+interface\s+(\w+)", "interface"),
    (r"^-\s*export\s+type\s+(\w+)\s*[={<]", "type"),
    (r"^-\s*export\s+(?:const\s+)?enum\s+(\w+)", "enum"),
]

# Patterns for Python declarations
PY_PATTERNS = [
    (r"^-\s*def\s+(\w+)\s*[\(\[]", "function"),
    (r"^-\s*async\s+def\s+(\w+)\s*[\(\[]", "function"),
    (r"^-\s*class\s+(\w+)[\s(:\[]", "class"),
    (r"^-\s*(_?[A-Z][A-Z_0-9]+)\s*[=:]", "const"),
]

SKIP_FILE_PATTERNS = [
    r"\.test\.",
    r"\.spec\.(ts|tsx|js|jsx)$",
    r"__tests__/",
    r"test/",
    r"tests/",
    r"\.stories\.",
    r"\.d\.ts$",
    r"__mocks__/",
    r"\.config\.",
    r"package\.json",
    r"package-lock\.json",
    r"tsconfig",
    r"\.cue$",
    r"\.go$",
    r"\.rs$",
]

# Patterns for state-machine import parsing of deleted lines
_IMPORT_OPEN_RE = re.compile(r"^-\s*import\s+(?:type\s+)?(?:\w+\s*,\s*)?\{")
_IMPORT_SINGLE_RE = re.compile(
    r"^-\s*import\s+(?:type\s+)?(?:\w+\s*,\s*)?\{([^}]+)\}\s+from\s+['\"]([^'\"]+)['\"]"
)
_IMPORT_DEFAULT_RE = re.compile(r"^-\s*import\s+(?:type\s+)?(\w+)\s+from\s+['\"]([^'\"]+)['\"]")
_IMPORT_FROM_RE = re.compile(r"from\s+['\"]([^'\"]+)['\"]")

SKIP_NAMES = {
    "default",
    "module",
    "exports",
    "require",
    "test",
    "describe",
    "it",
    "expect",
    "beforeEach",
    "afterEach",
    "beforeAll",
    "afterAll",
}


@dataclass
class RemovedDeclaration:
    file: str
    name: str
    type: str
    line_content: str


class DeadCodePlugin(EndpointPlugin):
    @property
    def name(self) -> str:
        return "dead_code"

    @property
    def api_path(self) -> str:
        return "/v1/analysis/dead-code"

    @property
    def baseline_prompt(self) -> str:
        return """You are an expert software architect. Find all dead code in this repository.

Analyze the codebase and identify ALL functions, classes, methods, and exported
constants that are dead code (never called, never imported, never referenced).

Focus on exported symbols -- functions, classes, and constants that are exported but
never imported or used anywhere in the codebase.

Do NOT include:
- Type definitions, interfaces, or enums (only runtime code)
- Test files or test utilities
- Entry points (main functions, CLI handlers, route handlers)
- Framework lifecycle hooks or decorators

CRITICAL: Update the existing REPORT.json file with your findings.
Format: a JSON object with "dead_code" array containing objects with file, name, type, and reason.
Set "analysis_complete" to true when done.
"""

    @property
    def enhanced_prompt(self) -> str:
        return """Read the file supermodel_dead_code_analysis.json in the current directory.

It contains a pre-computed static analysis. The deadCodeCandidates array lists functions
and classes that are exported but never imported or called anywhere in the codebase.

Filter out obvious false positives from the candidates:
- Framework lifecycle methods (execute, up, down, Template, etc.)
- Storybook stories and test utilities
- Classes loaded via dependency injection or plugin systems
- Database migration methods

CRITICAL: Update the existing REPORT.json file with your filtered findings.
Format: a JSON object with "dead_code" array containing objects with file, name, type, and reason.
Set "analysis_complete" to true when done.

Do NOT search the codebase. Just read the analysis file, filter, and update REPORT.json.
"""

    @property
    def enhanced_prompt_v2(self) -> str:
        return """You are an expert software architect. A static analyzer has pre-computed dead code
candidates for this codebase. Your job is to run a filter script and produce REPORT.json.

The file `supermodel_dead_code_analysis.json` in your working directory contains:
- `metadataSummary`: totalCandidates, rootFilesCount, reasonBreakdown, confidenceBreakdown
- `deadCodeCandidates`: all candidates (may be large — do NOT read the whole file manually)
- `entryPoints`: symbols confirmed alive — any candidate matching an entry point is a false positive

STEP 1: Run this Python script with Bash:

```python
import json

with open("supermodel_dead_code_analysis.json") as f:
    analysis = json.load(f)

summary = analysis.get("metadataSummary", {})
print(f"Total candidates: {summary.get('totalCandidates', '?')}, included: {summary.get('includedCandidates', '?')}")

# Build entry point whitelist
entry_set = {(ep.get("file", ""), ep.get("name", "")) for ep in analysis.get("entryPoints", [])}

# Filter candidates
dead_code = []
for c in analysis.get("deadCodeCandidates", []):
    key = (c.get("file", ""), c.get("name", ""))
    reason = c.get("reason", "")

    if key in entry_set:
        continue
    if "Type/interface" in reason:
        continue
    if c.get("confidence") not in ("high", None):
        continue

    dead_code.append({
        "file": c.get("file", ""),
        "name": c.get("name", ""),
        "type": c.get("type", "function"),
        "reason": reason,
    })

with open("REPORT.json", "w") as f:
    json.dump({"dead_code": dead_code, "analysis_complete": True}, f, indent=2)
print(f"Wrote {len(dead_code)} candidates to REPORT.json")
```

STEP 2: Verify REPORT.json was written by running: `python3 -c "import json; d=json.load(open('REPORT.json')); print(len(d['dead_code']), 'items written')"`

RULES:
- Do NOT read supermodel_dead_code_analysis.json manually — it may be very large.
- Do NOT grep or explore the codebase. Trust the pre-computed analysis.
- Run the script exactly as shown. Do not modify it.
- Type should be one of: function, class, method, const, interface, variable.
"""

    def parse_api_response(self, response: dict) -> dict:
        """Pre-filter the API response to remove obvious framework false positives.

        Args:
            response: Raw API response dict.
        """
        candidates = response.get("deadCodeCandidates", [])

        framework_names = re.compile(
            r"^(execute|up|down|Template|Story|stories|test|Test|Mock|mock|"
            r"Fixture|Spec|Suite|describe|it|expect|beforeEach|afterEach|"
            r"setUp|tearDown|default|module|exports|require)$"
        )
        framework_files = re.compile(
            r"(\.test\.|\.spec\.|\.stories\.|__tests__|__mocks__|"
            r"\.storybook|\.e2e\.|migrations/|\.d\.ts$)"
        )

        filtered = []
        for c in candidates:
            name = c.get("name", "")
            filepath = c.get("file", "")
            if framework_names.match(name):
                continue
            if framework_files.search(filepath):
                continue
            filtered.append(c)

        response = dict(response)
        response["deadCodeCandidates"] = filtered
        response["metadata"] = dict(response.get("metadata", {}))
        response["metadata"]["filteredCount"] = len(filtered)
        response["metadata"]["rawCount"] = len(candidates)
        return response

    def extract_ground_truth(
        self,
        repo: str,
        pr_number: int,
        language: str = "typescript",
        scope_prefix: str | None = None,
    ) -> list[dict]:
        diff = self.get_pr_diff(repo, pr_number)
        declarations = _parse_diff(diff, language)
        if scope_prefix:
            declarations = [d for d in declarations if d.file.startswith(scope_prefix)]

        # Filter feature-removal false positives: if a deleted symbol is imported
        # by another file also deleted in the same PR, it was live code (not dead
        # code) that was removed together with its consumers.  Such symbols should
        # not appear in the ground truth because no static-analysis tool would
        # ever report them as dead pre-merge.  (#714)
        deleted_imports = _parse_deleted_imports(diff)
        declarations = [d for d in declarations if not _is_feature_removal_fp(d, deleted_imports)]

        return [{"file": d.file, "name": d.name, "type": d.type} for d in declarations]


def _parse_diff(diff_text: str, language: str = "typescript") -> list[RemovedDeclaration]:
    patterns = TS_PATTERNS if language == "typescript" else PY_PATTERNS
    declarations = []
    current_file = None
    seen: set[tuple[str, str]] = set()

    for line in diff_text.split("\n"):
        if line.startswith("diff --git"):
            parts = line.split(" b/")
            if len(parts) >= 2:
                current_file = parts[-1]
            continue

        if not line.startswith("-") or line.startswith("---"):
            continue
        if current_file is None:
            continue
        if EndpointPlugin.should_skip_file(current_file, SKIP_FILE_PATTERNS):
            continue

        for pattern, decl_type in patterns:
            match = re.match(pattern, line)
            if match:
                name = match.group(1)
                if name in SKIP_NAMES:
                    continue
                key = (current_file, name)
                if key not in seen:
                    seen.add(key)
                    declarations.append(
                        RemovedDeclaration(
                            file=current_file,
                            name=name,
                            type=decl_type,
                            line_content=line.lstrip("-").strip(),
                        )
                    )
                break

    return declarations


def _parse_deleted_imports(diff_text: str) -> dict[str, set[str]]:
    """Parse deleted import statements, returning symbol → set[module_specifier].

    Handles single-line and multi-line named import blocks plus default imports.
    Capturing the module specifier (the `from '...'` part) lets callers do
    file-aware filtering rather than just name-based filtering.
    """
    # symbol_name -> set of module specifiers that imported it
    imports: dict[str, set[str]] = {}
    accumulating = False
    buf: list[str] = []

    def _add(name: str, spec: str) -> None:
        name = name.strip().split(" as ")[0].strip()
        if name:
            imports.setdefault(name, set()).add(spec)

    def _flush_block(raw: str) -> None:
        # Extract module specifier from the accumulated block
        m_from = _IMPORT_FROM_RE.search(raw)
        spec = m_from.group(1) if m_from else ""
        brace_open = raw.find("{")
        brace_close = raw.find("}")
        if brace_open != -1 and brace_close != -1:
            names_part = raw[brace_open + 1 : brace_close]
            for part in names_part.split(","):
                _add(part, spec)

    for line in diff_text.split("\n"):
        is_deleted = line.startswith("-") and not line.startswith("---")

        if not accumulating:
            if not is_deleted:
                continue
            # Single-line named import: -import { foo, bar } from '...'
            m = _IMPORT_SINGLE_RE.match(line)
            if m:
                spec = m.group(2)
                for part in m.group(1).split(","):
                    _add(part, spec)
                continue
            # Default import: -import Foo from '...'
            m = _IMPORT_DEFAULT_RE.match(line)
            if m:
                imports.setdefault(m.group(1), set()).add(m.group(2))
                continue
            # Start of a multi-line named import block
            if _IMPORT_OPEN_RE.match(line):
                accumulating = True
                buf = [line[1:]]  # strip leading '-'
                continue
        else:
            if is_deleted:
                buf.append(line[1:])
            joined = " ".join(buf)
            if "}" in joined:
                accumulating = False
                _flush_block(joined)
                buf = []

    return imports


def _is_feature_removal_fp(
    decl: "RemovedDeclaration", deleted_imports: dict[str, set[str]]
) -> bool:
    """True if the declaration appears to be a feature-removal false positive.

    A symbol is considered a false positive when another deleted file in the
    same PR imports it FROM a module whose basename matches the declaration's
    file. This ties the filter to the actual source file rather than just the
    name, preventing spurious suppression of unrelated same-named exports.
    """
    if decl.name not in deleted_imports:
        return False
    decl_stem = re.sub(r"\.(ts|tsx|js|jsx)$", "", decl.file.split("/")[-1])
    for spec in deleted_imports[decl.name]:
        spec_stem = re.sub(r"\.(ts|tsx|js|jsx)$", "", spec.rstrip("/").split("/")[-1])
        if spec_stem and spec_stem == decl_stem:
            return True
    return False
