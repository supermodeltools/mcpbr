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
    (r"^-\s*export\s+interface\s+(\w+)", "interface"),
    (r"^-\s*export\s+type\s+(\w+)\s*[=<{]", "type"),
    (r"^-\s*export\s+enum\s+(\w+)", "enum"),
    (r"^-\s*export\s+default\s+(?:async\s+)?function\s+(\w+)", "function"),
    (r"^-\s*export\s+default\s+class\s+(\w+)", "class"),
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
candidates for this codebase. Your job is to FILTER them using the metadata provided.

STEP 1: Read `supermodel_dead_code_analysis.json`. It contains:
- `metadataSummary`: totalCandidates, rootFilesCount, reasonBreakdown, confidenceBreakdown
- `chunkFiles`: list of chunk files with candidate details
- `entryPoints`: symbols confirmed alive — any candidate matching an entry point is a false positive

If there are chunk files, read ALL of them.

STEP 2: Understand the analysis quality.
- Check `rootFilesCount` — if it's much higher than expected (>20), the import
  resolver likely failed on many files, meaning "file never imported" candidates
  have a high false positive rate for framework-wired code.
- Check `reasonBreakdown` to understand where candidates come from.

STEP 3: Write a script to filter candidates and produce REPORT.json:

```python
import json, glob

with open("supermodel_dead_code_analysis.json") as f:
    index = json.load(f)

# Load entry points as a whitelist
entry_set = set()
for ep in index.get("entryPoints", []):
    entry_set.add((ep.get("file", ""), ep.get("name", "")))

# Load all candidates from chunk files
candidates = []
for chunk_ref in index.get("chunkFiles", []):
    with open(chunk_ref["file"]) as f:
        chunk = json.load(f)
    candidates.extend(chunk.get("deadCodeCandidates", []))

# Filter
dead_code = []
for c in candidates:
    key = (c.get("file", ""), c.get("name", ""))
    reason = c.get("reason", "")
    confidence = c.get("confidence", "")

    # Drop entry points
    if key in entry_set:
        continue

    # Drop low-signal reasons
    if "Type/interface" in reason or "no references" in reason:
        continue

    # Keep everything else — the graph already did import/call analysis
    dead_code.append({
        "file": c.get("file", ""),
        "name": c.get("name", ""),
        "type": c.get("type", "function"),
        "reason": reason
    })

with open("REPORT.json", "w") as f:
    json.dump({"dead_code": dead_code, "analysis_complete": True}, f, indent=2)
print(f"Wrote {len(dead_code)} candidates to REPORT.json")
```

STEP 4: Run the script, then read REPORT.json to confirm it was written correctly.

RULES:
- Do NOT grep the codebase to verify candidates. The static analyzer already
  performed call graph and dependency analysis — grep produces false negatives
  when symbol names appear in comments, strings, or type-only imports.
- Trust the graph. Filter only using the metadata (reason, confidence, entryPoints).
- Type should be one of: function, class, method, const, interface, variable.
- When in doubt about a candidate, INCLUDE it — missing real dead code is worse
  than a false positive.
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
