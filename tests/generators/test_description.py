import os

import pytest

TEST_LLM_MODEL_PATH = "microsoft/Phi-3-mini-128k-instruct"


@pytest.mark.skipif(os.getenv("USE_LLM", "FALSE").lower() != "true", reason="Skipping LLM tests")
class TestDescriptionGenerator:
    def setup_method(self) -> None:
        from qxmt.generators.description import DescriptionGenerator

        self.generator = DescriptionGenerator(model_path=TEST_LLM_MODEL_PATH)

    def test__create_message(self) -> None:
        message = self.generator._create_message(
            system_prompt="system_prompt",
            user_prompt="user_prompt",
        )

        expected_message = [
            {"role": "system", "content": "system_prompt"},
            {"role": "user", "content": "user_prompt"},
        ]

        assert message == expected_message

    def test_generate(self) -> None:
        add_code = "add code: test"
        remove_code = "remove code: test"
        output = self.generator.generate(add_code=add_code, remove_code=remove_code)

        assert isinstance(output, str)

    def test_generate_no_code_changes(self) -> None:
        add_code = ""
        remove_code = ""
        output = self.generator.generate(add_code=add_code, remove_code=remove_code)

        assert output == "No code changes detected on local git repository."
