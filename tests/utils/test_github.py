import subprocess

from pytest_mock import MockerFixture

from qxmt.constants import MODULE_HOME
from qxmt.logger import set_default_logger
from qxmt.utils.github import (
    get_commit_id,
    get_git_add_code,
    get_git_diff,
    get_git_rm_code,
)

LOGGER = set_default_logger(__name__)


class TestGetCommitId:
    # success to get commit id
    def test_get_commit_id_success(self, mocker: MockerFixture) -> None:
        mock_check_output = mocker.patch("subprocess.check_output")
        mock_check_output.return_value = b"dummy_commit_id\n"

        commit_id = get_commit_id(repo_path=MODULE_HOME)

        assert commit_id == "dummy_commit_id"
        mock_check_output.assert_called_once_with(["git", "-C", str(MODULE_HOME), "rev-parse", "HEAD"])

    # failure to get commit id
    def test_get_commit_id_not_git_repo(self, mocker: MockerFixture) -> None:
        mocker.patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "git"))

        commit_id = get_commit_id(repo_path=MODULE_HOME)
        assert commit_id == ""


class TestGetGitDiff:
    # success to get git diff
    def test_get_git_diff_success(self, mocker: MockerFixture) -> None:
        mock_check_output = mocker.patch("subprocess.check_output")
        mock_check_output.return_value = b"dummy_diff_output\n"

        diff = get_git_diff(repo_path=MODULE_HOME)

        assert diff == "dummy_diff_output"
        mock_check_output.assert_called_once_with(["git", "-C", str(MODULE_HOME), "diff"])

    # failure to get git diff
    def test_get_git_diff_not_git_repo(self, mocker: MockerFixture) -> None:
        mocker.patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "git"))

        diff = get_git_diff(repo_path=MODULE_HOME)
        assert diff == ""


class TestGetGitAdd:
    def test_get_git_add_code_success(self, mocker: MockerFixture) -> None:
        diff = "+++test.py\n+import subprocess\n+import pytest\n+from pytest_mock import MockerFixture\n"
        mocker.patch("qxmt.utils.github.get_git_diff", return_value=diff)

        # set diff parameter
        added_code = get_git_add_code(diff)
        assert added_code == "import subprocess\nimport pytest\nfrom pytest_mock import MockerFixture\n"

        # not set diff parameter
        added_code = get_git_add_code()
        assert added_code == "import subprocess\nimport pytest\nfrom pytest_mock import MockerFixture\n"


class TestGetGitRm:
    def test_get_git_rm_code_success(self, mocker: MockerFixture) -> None:
        diff = "---test.py\n-import subprocess\n-import pytest\n-import pytest_mock\n"
        mocker.patch("qxmt.utils.github.get_git_diff", return_value=diff)

        # set diff parameter
        rm_code = get_git_rm_code(diff)
        assert rm_code == "import subprocess\nimport pytest\nimport pytest_mock\n"

        # not set diff parameter
        rm_code = get_git_rm_code()
        assert rm_code == "import subprocess\nimport pytest\nimport pytest_mock\n"
