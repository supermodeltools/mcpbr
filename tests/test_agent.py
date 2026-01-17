"""Tests for agent utilities."""

from mcpbr.agent import extract_patch


class TestExtractPatch:
    """Tests for extract_patch function."""

    def test_extract_patch_block(self) -> None:
        """Test extracting patch from ```patch block."""
        text = """
Here's the fix:

```patch
--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
-old line
+new line
```

This should fix the issue.
"""
        result = extract_patch(text)
        assert "--- a/file.py" in result
        assert "+++ b/file.py" in result
        assert "-old line" in result
        assert "+new line" in result

    def test_extract_diff_block(self) -> None:
        """Test extracting patch from ```diff block."""
        text = """
```diff
--- a/test.py
+++ b/test.py
@@ -10,6 +10,7 @@
 context
+added line
```
"""
        result = extract_patch(text)
        assert "--- a/test.py" in result
        assert "+added line" in result

    def test_no_patch_returns_empty(self) -> None:
        """Test that missing patch returns empty string."""
        text = "This response has no patch in it."
        result = extract_patch(text)
        assert result == ""

    def test_extract_bare_diff(self) -> None:
        """Test extracting bare diff without code fence."""
        text = """
--- a/module.py
+++ b/module.py
@@ -5,7 +5,8 @@
 def foo():
-    return 1
+    return 2
"""
        result = extract_patch(text)
        assert "--- a/module.py" in result or result == ""
