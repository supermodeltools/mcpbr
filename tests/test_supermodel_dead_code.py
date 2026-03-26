"""Tests for the dead code endpoint ground truth extraction and FP filtering.

Regression tests for issue #714: symbols from entirely deleted files that were
actively imported by other (also deleted) files are feature-removal FPs and
must be excluded from ground truth.
"""

from mcpbr.benchmarks.supermodel.endpoints.dead_code import (
    RemovedDeclaration,
    _is_feature_removal_fp,
    _parse_deleted_imports,
    _parse_diff,
)

# ---------------------------------------------------------------------------
# Fixtures — synthetic diffs representative of the benchmark PRs in issue #714
# ---------------------------------------------------------------------------

# n8n-io/n8n PR #23572: removes Pyodide-related files entirely.
# PythonSandbox is imported by another deleted file, so it is a feature-removal FP.
# LoadPyodide has no deleted importer — it was genuinely orphaned dead code.
N8N_PYODIDE_DIFF = """\
diff --git a/packages/nodes-base/nodes/Code/Pyodide.ts b/packages/nodes-base/nodes/Code/Pyodide.ts
deleted file mode 100644
--- a/packages/nodes-base/nodes/Code/Pyodide.ts
+++ /dev/null
@@ -1,10 +0,0 @@
-export async function LoadPyodide() {
-  return await loadPyodide();
-}
diff --git a/packages/nodes-base/nodes/Code/PythonSandbox.ts b/packages/nodes-base/nodes/Code/PythonSandbox.ts
deleted file mode 100644
--- a/packages/nodes-base/nodes/Code/PythonSandbox.ts
+++ /dev/null
@@ -1,10 +0,0 @@
-export class PythonSandbox {
-  run() {}
-}
diff --git a/packages/nodes-base/nodes/Code/CodeRunner.ts b/packages/nodes-base/nodes/Code/CodeRunner.ts
--- a/packages/nodes-base/nodes/Code/CodeRunner.ts
+++ b/packages/nodes-base/nodes/Code/CodeRunner.ts
@@ -1,5 +1,2 @@
-import { PythonSandbox } from './PythonSandbox';
 export function runCode() {}
"""

# prisma/prisma PR #28485: removes extractSqliteSources.ts and serializeDatasources.ts.
# serializeDatasources is imported by extractSqliteSources (both deleted) → FP.
# extractSqliteSources has no deleted importer → genuine dead code.
PRISMA_DIFF = """\
diff --git a/packages/client-generator-js/src/extractSqliteSources.ts b/packages/client-generator-js/src/extractSqliteSources.ts
deleted file mode 100644
--- a/packages/client-generator-js/src/extractSqliteSources.ts
+++ /dev/null
@@ -1,8 +0,0 @@
-import { serializeDatasources } from './serializeDatasources';
-export function extractSqliteSources() {}
-export interface DatasourceOverwrite {
-  url: string;
-}
diff --git a/packages/client-generator-js/src/serializeDatasources.ts b/packages/client-generator-js/src/serializeDatasources.ts
deleted file mode 100644
--- a/packages/client-generator-js/src/serializeDatasources.ts
+++ /dev/null
@@ -1,5 +0,0 @@
-export function serializeDatasources() {}
-export function datasourceToDatasourceOverwrite() {}
"""

# Simple diff where nothing is imported — all deleted symbols are genuine dead code.
ALL_GENUINELY_DEAD_DIFF = """\
diff --git a/src/utils/legacy.ts b/src/utils/legacy.ts
deleted file mode 100644
--- a/src/utils/legacy.ts
+++ /dev/null
@@ -1,6 +0,0 @@
-export function oldHelper() {}
-export const DEPRECATED_CONSTANT = 'x';
-export interface LegacyConfig {
-  key: string;
-}
"""

# Diff where a multi-line import block is also deleted.
MULTILINE_IMPORT_DIFF = """\
diff --git a/src/module.ts b/src/module.ts
--- a/src/module.ts
+++ b/src/module.ts
@@ -1,8 +1,2 @@
-import {
-  removedFn,
-  anotherFn,
-} from './removed';
 export function survivor() {}
diff --git a/src/removed.ts b/src/removed.ts
deleted file mode 100644
--- a/src/removed.ts
+++ /dev/null
@@ -1,5 +0,0 @@
-export function removedFn() {}
-export function anotherFn() {}
-export function orphanFn() {}
"""


# ---------------------------------------------------------------------------
# _parse_deleted_imports
# ---------------------------------------------------------------------------


class TestParseDeletedImports:
    def test_single_line_named_import(self) -> None:
        imports = _parse_deleted_imports(N8N_PYODIDE_DIFF)
        assert "PythonSandbox" in imports
        assert any("PythonSandbox" in spec for spec in imports["PythonSandbox"])

    def test_no_deleted_imports_for_load_pyodide(self) -> None:
        imports = _parse_deleted_imports(N8N_PYODIDE_DIFF)
        assert "LoadPyodide" not in imports

    def test_single_line_named_import_prisma(self) -> None:
        imports = _parse_deleted_imports(PRISMA_DIFF)
        assert "serializeDatasources" in imports

    def test_multiline_import_block(self) -> None:
        imports = _parse_deleted_imports(MULTILINE_IMPORT_DIFF)
        assert "removedFn" in imports
        assert "anotherFn" in imports

    def test_empty_diff_returns_empty(self) -> None:
        assert _parse_deleted_imports("") == {}


# ---------------------------------------------------------------------------
# _is_feature_removal_fp
# ---------------------------------------------------------------------------


class TestIsFeatureRemovalFp:
    def _decl(self, file: str, name: str) -> RemovedDeclaration:
        return RemovedDeclaration(file=file, name=name, type="function", line_content="")

    def test_python_sandbox_is_fp(self) -> None:
        imports = _parse_deleted_imports(N8N_PYODIDE_DIFF)
        decl = self._decl("packages/nodes-base/nodes/Code/PythonSandbox.ts", "PythonSandbox")
        assert _is_feature_removal_fp(decl, imports) is True

    def test_load_pyodide_is_not_fp(self) -> None:
        imports = _parse_deleted_imports(N8N_PYODIDE_DIFF)
        decl = self._decl("packages/nodes-base/nodes/Code/Pyodide.ts", "LoadPyodide")
        assert _is_feature_removal_fp(decl, imports) is False

    def test_serialize_datasources_is_fp(self) -> None:
        imports = _parse_deleted_imports(PRISMA_DIFF)
        decl = self._decl(
            "packages/client-generator-js/src/serializeDatasources.ts",
            "serializeDatasources",
        )
        assert _is_feature_removal_fp(decl, imports) is True

    def test_extract_sqlite_sources_is_not_fp(self) -> None:
        imports = _parse_deleted_imports(PRISMA_DIFF)
        decl = self._decl(
            "packages/client-generator-js/src/extractSqliteSources.ts",
            "extractSqliteSources",
        )
        assert _is_feature_removal_fp(decl, imports) is False

    def test_multiline_import_symbols_are_fp(self) -> None:
        imports = _parse_deleted_imports(MULTILINE_IMPORT_DIFF)
        for name in ("removedFn", "anotherFn"):
            decl = self._decl("src/removed.ts", name)
            assert _is_feature_removal_fp(decl, imports) is True

    def test_orphan_fn_is_not_fp(self) -> None:
        imports = _parse_deleted_imports(MULTILINE_IMPORT_DIFF)
        decl = self._decl("src/removed.ts", "orphanFn")
        assert _is_feature_removal_fp(decl, imports) is False

    def test_all_genuinely_dead_are_not_fp(self) -> None:
        imports = _parse_deleted_imports(ALL_GENUINELY_DEAD_DIFF)
        for name in ("oldHelper", "DEPRECATED_CONSTANT", "LegacyConfig"):
            decl = self._decl("src/utils/legacy.ts", name)
            assert _is_feature_removal_fp(decl, imports) is False


# ---------------------------------------------------------------------------
# _parse_diff + FP filter integration (mirrors extract_ground_truth logic)
# ---------------------------------------------------------------------------


class TestGroundTruthFiltering:
    """Integration: parse diff then apply FP filter — simulates extract_ground_truth."""

    def _filtered_gt(self, diff: str) -> list[str]:
        decls = _parse_diff(diff)
        deleted_imports = _parse_deleted_imports(diff)
        surviving = [d for d in decls if not _is_feature_removal_fp(d, deleted_imports)]
        return [d.name for d in surviving]

    def test_n8n_keeps_load_pyodide_drops_python_sandbox(self) -> None:
        # LoadPyodide — no deleted importer → genuine dead code, kept in GT
        # PythonSandbox — imported by deleted CodeRunner.ts → FP, removed from GT
        names = self._filtered_gt(N8N_PYODIDE_DIFF)
        assert "LoadPyodide" in names
        assert "PythonSandbox" not in names

    def test_prisma_keeps_extract_drops_serialize(self) -> None:
        # extractSqliteSources has no deleted importer → kept
        # serializeDatasources imported by deleted extractSqliteSources.ts → FP, dropped
        # datasourceToDatasourceOverwrite has no deleted importer → kept
        # DatasourceOverwrite has no deleted importer → kept
        names = self._filtered_gt(PRISMA_DIFF)
        assert "extractSqliteSources" in names
        assert "DatasourceOverwrite" in names
        assert "datasourceToDatasourceOverwrite" in names
        assert "serializeDatasources" not in names

    def test_all_genuinely_dead_all_kept(self) -> None:
        names = self._filtered_gt(ALL_GENUINELY_DEAD_DIFF)
        assert set(names) == {"oldHelper", "DEPRECATED_CONSTANT", "LegacyConfig"}

    def test_multiline_import_orphan_kept_others_dropped(self) -> None:
        names = self._filtered_gt(MULTILINE_IMPORT_DIFF)
        assert "orphanFn" in names
        assert "removedFn" not in names
        assert "anotherFn" not in names
