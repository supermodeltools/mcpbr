"""Tests for evaluation logic."""

from mcpbr.evaluation import parse_test_list


class TestParseTestList:
    """Tests for parse_test_list function."""

    def test_parse_json_list(self) -> None:
        """Test parsing JSON format test list."""
        result = parse_test_list('["test_foo.py", "test_bar.py::test_baz"]')
        assert result == ["test_foo.py", "test_bar.py::test_baz"]

    def test_parse_python_literal(self) -> None:
        """Test parsing Python literal format."""
        result = parse_test_list("['test_one', 'test_two']")
        assert result == ["test_one", "test_two"]

    def test_parse_empty_string(self) -> None:
        """Test parsing empty string."""
        result = parse_test_list("")
        assert result == []

    def test_parse_empty_list(self) -> None:
        """Test parsing empty list."""
        result = parse_test_list("[]")
        assert result == []

    def test_parse_single_item(self) -> None:
        """Test parsing single item list."""
        result = parse_test_list('["single_test"]')
        assert result == ["single_test"]
