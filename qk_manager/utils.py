from pathlib import Path

from qk_manager.exceptions import InvalidFileExtensionError


def check_json_extension(file_path: str | Path) -> None:
    if Path(file_path).suffix.lower() != ".json":
        raise InvalidFileExtensionError(f"File '{file_path}' does not have a '.json' extension.")
