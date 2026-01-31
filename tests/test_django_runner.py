"""Tests for Django test runner detection and command building."""

from mcpbr.evaluation import _build_test_command


class TestDjangoRunner:
    """Test Django test runner detection and command building."""

    def test_django_format_with_parentheses(self):
        """Test Django format: test_method (module.TestClass)"""
        test = "test_to_python (annotations.tests.SimpleTestCase)"
        cmd = _build_test_command(test, uses_prebuilt=True)
        assert "cd /testbed/tests && ./runtests.py annotations.tests" in cmd
        assert "conda activate testbed" in cmd

    def test_django_format_dot_separated(self):
        """Test Django format: dot-separated module path"""
        test = "annotations.tests.test_method"
        cmd = _build_test_command(test, uses_prebuilt=True)
        assert "cd /testbed/tests && ./runtests.py annotations.tests" in cmd

    def test_django_format_without_prebuilt(self):
        """Test Django format without prebuilt environment"""
        test = "test_method (admin_views.tests.TestCase)"
        cmd = _build_test_command(test, uses_prebuilt=False)
        assert "cd /testbed/tests && ./runtests.py admin_views.tests" in cmd
        assert "conda" not in cmd

    def test_pytest_format_with_colons(self):
        """Test pytest format with :: separator"""
        test = "tests/test_file.py::test_function"
        cmd = _build_test_command(test, uses_prebuilt=False)
        assert "python -m pytest" in cmd
        assert "::" in cmd
        assert "runtests.py" not in cmd

    def test_pytest_format_file_path(self):
        """Test pytest format with just file path"""
        test = "tests/test_file.py"
        cmd = _build_test_command(test, uses_prebuilt=False)
        assert "python -m pytest" in cmd
        assert "tests/test_file.py" in cmd
        assert "runtests.py" not in cmd

    def test_pytest_format_with_k_flag(self):
        """Test pytest format requiring -k flag"""
        test = "test_function_name"
        cmd = _build_test_command(test, uses_prebuilt=False)
        assert "python -m pytest -k" in cmd
        assert "test_function_name" in cmd
        assert "runtests.py" not in cmd

    def test_absolute_path_is_pytest(self):
        """Test that absolute paths use pytest, not Django runner"""
        test = "/testbed/tests/test_file.py"
        cmd = _build_test_command(test, uses_prebuilt=False)
        assert "python -m pytest" in cmd
        assert "runtests.py" not in cmd

    def test_django_complex_module_path(self):
        """Test Django format with complex module path"""
        test = "test_method (db.models.fields.tests.TestCase)"
        cmd = _build_test_command(test, uses_prebuilt=True)
        assert "cd /testbed/tests && ./runtests.py db.models" in cmd

    def test_prebuilt_environment_activation(self):
        """Test that prebuilt images get conda activation"""
        test = "test_simple"
        cmd = _build_test_command(test, uses_prebuilt=True)
        assert "source /opt/miniconda3/etc/profile.d/conda.sh" in cmd
        assert "conda activate testbed" in cmd

    def test_no_prebuilt_environment_activation(self):
        """Test that non-prebuilt images don't get conda activation"""
        test = "test_simple"
        cmd = _build_test_command(test, uses_prebuilt=False)
        assert "conda" not in cmd
        assert "source /opt/miniconda3" not in cmd
