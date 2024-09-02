from qxmt.utils import __all__

EXPECTED_ALL = [
    "get_commit_id",
    "get_number_of_qubits",
    "get_git_diff",
    "get_git_add_code",
    "get_git_rm_code",
    "get_platform_from_device",
    "load_yaml_config",
    "load_object_from_yaml",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
