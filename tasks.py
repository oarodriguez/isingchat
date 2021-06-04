"""
Collection of development tasks.

Usage:
    python -m tasks TASK-NAME
"""
from pathlib import Path
from subprocess import run
from typing import List

import click

from isingchat import __version__

PROJECT_DIR = Path(__file__).parent
SRC_DIR = PROJECT_DIR / "src"
TESTS_DIR = PROJECT_DIR / "tests"
DOCS_DIR = PROJECT_DIR / "docs"
DOCS_SOURCE_DIR = DOCS_DIR / "source"
DOCS_BUILD_DIR = DOCS_DIR / "build"
TASKS_FILE = PROJECT_DIR / "tasks.py"

ISORT_CMD = "isort"
BLACK_CMD = "black"
PYDOCSTYLE_CMD = "pydocstyle"
FLAKE8_CMD = "flake8"
MYPY_CMD = "mypy"
PYTEST_CMD = "pytest"
SPHINX_BUILD_CMD = "sphinx-build"

# Coverage report XML file.
COVERAGE_XML = "coverage.xml"

# Arguments to pass to subprocess.run for each task.
BUILD_DOCS_ARGS = [
    SPHINX_BUILD_CMD,
    str(DOCS_SOURCE_DIR),
    str(DOCS_BUILD_DIR),
]
FORMAT_ARGS = [
    BLACK_CMD,
    str(TASKS_FILE),
    str(SRC_DIR),
    str(TESTS_DIR),
    str(DOCS_DIR),
]
ISORT_ARGS = [
    ISORT_CMD,
    str(TASKS_FILE),
    str(SRC_DIR),
    str(TESTS_DIR),
]
PYDOCSTYLE_ARGS = [
    PYDOCSTYLE_CMD,
    str(TASKS_FILE),
    str(SRC_DIR),
    str(TESTS_DIR),
]
FLAKE8_ARGS = [
    FLAKE8_CMD,
    str(TASKS_FILE),
    str(SRC_DIR),
    str(TESTS_DIR),
    "--statistics",
]
PYTEST_ARGS = [
    PYTEST_CMD,
    "--cov",
    "--cov-report",
    "term-missing",
    "--cov-report",
    f"xml:./{COVERAGE_XML}",
]
MYPY_ARGS = [
    MYPY_CMD,
    str(TASKS_FILE),
    str(SRC_DIR),
]


def _run(command: List[str]):
    """Run a subcommand through python subprocess.run routine."""
    # NOTE: See https://stackoverflow.com/a/32799942 in case we want to
    #  remove shell=True.
    run(command)


# Main entry point.
app = click.Group("tasks")


@app.command()
def black():
    """Format the source code using black."""
    _run(FORMAT_ARGS)


@app.command()
def isort():
    """Format imports using isort."""
    _run(ISORT_ARGS)


@app.command()
def pydocstyle():
    """Run pydocstyle."""
    _run(PYDOCSTYLE_ARGS)


@app.command()
def mypy():
    """Run mypy."""
    _run(MYPY_ARGS)


@app.command()
def tests():
    """Run test suite."""
    _run(PYTEST_ARGS)


@app.command()
def version():
    """Run mypy."""
    print(__version__)


@app.command(name="format")
def format_():
    """Run all formatting tasks."""
    _run(FORMAT_ARGS)
    _run(ISORT_ARGS)


@app.command()
def format_check():
    """Run all formatting and typechecking tasks."""
    _run(FORMAT_ARGS)
    _run(ISORT_ARGS)
    _run(MYPY_ARGS)


@app.command()
def lint():
    """Run all linting tasks."""
    _run(PYDOCSTYLE_ARGS)
    _run(FLAKE8_ARGS)


@app.command()
@click.option(
    "-f",
    "--doc-format",
    type=click.Choice(["html"]),
    default="html",
    help="Generated documentation format",
)
def build_docs(doc_format: str):
    """Build the documentation."""
    BUILD_DOCS_ARGS.extend(["-b", doc_format])
    _run(BUILD_DOCS_ARGS)


if __name__ == "__main__":
    app(prog_name="tasks")
