import argparse
from pathlib import Path

import pretty_errors
import torch
from peft.peft_model import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from train.src.config import Config, ModelConfig


class PeftMerger:
    def __init__(
        self,
        config_path: str,
        push_to_hub: bool = False,
        root_folder: str = "",
    ):
        self.push_to_hub = push_to_hub
        self.config = Config.load_config(config_path)
        self.model_config = ModelConfig(**self.config.pop("model"))
        self.root_folder = root_folder

    def load_models(self):
        print(f"Loading base model: {self.model_config.base_model}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_config.base_model,
            torch_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.base_model)
        self.base_model.resize_token_embeddings(len(self.tokenizer))

        print(f"Loading PEFT: {self.model_config.new_model}")
        path_model_path = str(Path(self.root_folder, self.model_config.new_model))
        self.peft_model = PeftModel.from_pretrained(self.base_model, path_model_path)

    def merge_and_unload(self):
        print(f"Running merge_and_unload")
        self.peft_model = self.peft_model.merge_and_unload()

    def save_models(self):
        path_str = str(Path(self.root_folder, f"{self.model_config.new_model}_merged"))
        self.peft_model.save_pretrained(path_str)
        self.tokenizer.save_pretrained(path_str)
        print(f"Model saved to {path_str}")
        if self.push_to_hub:
            print(f"Saving to hub ...")
            self.peft_model.push_to_hub(self.model_config.new_model)
            self.tokenizer.push_to_hub(self.model_config.new_model)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--root_folder", type=str, default=None)
    parser.add_argument("--push_to_hub", action=argparse.BooleanOptionalAction)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    peft_merger = PeftMerger(
        config_path=args.config_path,
        push_to_hub=args.push_to_hub,
        root_folder=args.root_folder,
    )
    peft_merger.load_models()
    peft_merger.merge_and_unload()
    peft_merger.save_models()
