"""Test command specs from upstream SWE-bench harness.

Maps repositories to their correct test commands. mcpbr defaults to pytest
for all non-Django projects, but some projects (e.g., sympy) use custom test
runners that aren't pytest-compatible.

Source: https://github.com/SWE-bench/SWE-bench/blob/main/swebench/harness/constants/python.py
"""

# Base test commands per framework (from upstream constants/python.py)
TEST_PYTEST = "pytest --no-header -rA --tb=no -p no:cacheprovider"
TEST_DJANGO = "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1"
TEST_SYMPY = "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose"
TEST_SPHINX = "tox --current-env -epy39 -v --"
TEST_ASTROPY = "pytest -rA -vv -o console_output_style=classic --tb=no"
TEST_SEABORN = "pytest --no-header -rA"

# Repo → test command mapping
# Only non-pytest entries need to be here — pytest is the default fallback.
# Django is included for documentation but its existing handler takes precedence.
REPO_TO_TEST_CMD: dict[str, str] = {
    "sympy/sympy": TEST_SYMPY,
    "django/django": TEST_DJANGO,
    "sphinx-doc/sphinx": TEST_SPHINX,
}


def get_repo_test_command(repo: str) -> str | None:
    """Look up the upstream test command for a repo.

    Returns None if repo uses standard pytest (handled by existing logic).
    """
    return REPO_TO_TEST_CMD.get(repo)


# Language → default test command for SWE-bench Pro multi-language support
LANGUAGE_TO_TEST_CMD: dict[str, str] = {
    "python": TEST_PYTEST,
    "go": "go test -v -count=1",
    "typescript": "npx jest --verbose --no-cache",
    "javascript": "npx jest --verbose --no-cache",
}


def get_language_test_command(language: str) -> str | None:
    """Look up the default test command for a programming language.

    Args:
        language: Programming language name (lowercase).

    Returns:
        Default test command string, or None if language is not recognized.
    """
    return LANGUAGE_TO_TEST_CMD.get(language.lower())
