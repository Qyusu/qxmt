import subprocess
from logging import Logger
from pathlib import Path
from typing import Optional

from qxmt.constants import MODULE_HOME
from qxmt.logger import set_default_logger

LOGGER = set_default_logger(__name__)


def get_commit_id(repo_path: Path | str = MODULE_HOME, logger: Logger = LOGGER) -> str:
    """Get the commit ID of the current git repository.
    if the current directory is not a git repository, return an empty string.

    Args:
        repo_path (Path | str): git repository path. Defaults to MODULE_HOME.
        logger (Logger): logger. Defaults to LOGGER.

    Returns:
        str: git commit ID
    """
    command = ["git", "-C", str(repo_path), "rev-parse", "HEAD"]
    try:
        commit_id = subprocess.check_output(command).strip().decode("utf-8")
        return commit_id
    except subprocess.CalledProcessError as e:
        command_str = " ".join(command)
        logger.warning(f'Error executing "{command_str}" command: {e}')
        return ""


def get_git_diff(repo_path: Path | str = MODULE_HOME, logger: Logger = LOGGER) -> str:
    """Get the git diff of the current git repository.
    if the current directory is not a git repository, return an empty string.

    Args:
        repo_path (Path | str): git repository path. Defaults to MODULE_HOME.
        logger (Logger): logger. Defaults to LOGGER.

    Returns:
        str: git diff
    """
    command = ["git", "-C", str(repo_path), "diff"]
    try:
        diff = subprocess.check_output(command).strip().decode("utf-8")
        return diff
    except subprocess.CalledProcessError as e:
        command_str = " ".join(command)
        logger.warning(f'Error executing "{command_str}" command: {e}')
        return ""


def get_git_add_code(
    diff: Optional[str] = None,
    repo_path: Path | str = MODULE_HOME,
    logger: Logger = LOGGER,
) -> str:
    """Get the added code from the git diff.

    Args:
        diff (Optional[str]): string of git diff
        repo_path (Path | str): git repository path. Defaults to MODULE_HOME.
        logger (Logger): logger. Defaults to LOGGER.

    Returns:
        str: added code
    """
    if diff is None:
        diff = get_git_diff(repo_path=repo_path, logger=logger)

    added_code = ""
    for line in diff.split("\n"):
        # ignore the line that starts with "+++" because it is the file
        if line.startswith("+") and not line.startswith("+++"):
            added_code += line[1:] + "\n"

    return added_code


def get_git_rm_code(
    diff: Optional[str] = None,
    repo_path: Path | str = MODULE_HOME,
    logger: Logger = LOGGER,
) -> str:
    """Get the removed code from the git diff.

    Args:
        diff (Optional[str]): string of git diff
        repo_path (Path | str): git repository path. Defaults to MODULE_HOME.
        logger (Logger): logger. Defaults to LOGGER.

    Returns:
        str: removed code
    """
    if diff is None:
        diff = get_git_diff(repo_path=repo_path, logger=logger)

    removed_code = ""
    for line in diff.split("\n"):
        # ignore the line that starts with "---" because it is the file
        if line.startswith("-") and not line.startswith("---"):
            removed_code += line[1:] + "\n"

    return removed_code
