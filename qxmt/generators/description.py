import os

from qxmt.constants import LLM_MODEL_PATH
from qxmt.generators.prompts import diff_desc_system_prompt, diff_desc_user_prompt

if os.getenv("USE_LLM", "FALSE").lower() == "true":
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer


class DescriptionGenerator:
    def __init__(self, model_path: str = LLM_MODEL_PATH) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

    def _create_message(self, system_prompt: str, user_prompt: str) -> list[dict]:
        """Create a message for the model input.

        Args:
            system_prompt (str): system prompt
            user_prompt (str): user prompt

        Returns:
            list[dict]: message for the model input
        """
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return message

    def generate(
        self,
        add_code: str = "",
        remove_code: str = "",
        system_prompt: str = diff_desc_system_prompt,
        user_prompt: str = diff_desc_user_prompt,
    ) -> str:
        """Generate a description for the input diff code.

        Args:
            add_code (str, optional): added code. Defaults to "".
            remove_code (str, optional): removed code. Defaults to "".
            system_prompt (str, optional): system prompt. Defaults to diff_desc_system_prompt.
            user_prompt (str, optional): user prompt. Defaults to diff_desc_user_prompt.

        Returns:
            str: generated description
        """
        if add_code == "" and remove_code == "":
            return "No code changes detected on local git repository."

        user_prompt = user_prompt.format(add_code=add_code, remove_code=remove_code)
        message = self._create_message(system_prompt, user_prompt)
        prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        token_ids: torch.Tensor = self.tokenizer.encode(
            str(prompt),
            add_special_tokens=False,
            return_tensors="pt",
        )  # type: ignore

        with torch.no_grad():
            output_ids = self.model.generate(
                token_ids.to(device=self.model.device),
                do_sample=None,
                temperature=0.0,
                max_new_tokens=256,
            )
        output = self.tokenizer.decode(output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True)

        return output
