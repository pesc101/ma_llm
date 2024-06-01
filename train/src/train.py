import argparse
import sys

import pretty_errors
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer

from shared.src.utils.print import print_rich
from train.src.config import (
    BitsAndBytesConfigImporter,
    Config,
    DatasetConfig,
    LoRAConfigImporter,
    ModelConfig,
    SFTTConfig,
    TrainingArgumentsConfigImporter,
)


class Trainer:
    def __init__(self, config_path: str, wandb: bool = False):
        self.config_path = config_path
        self.wandb = wandb

    def load_configs(self):
        print_rich("Load Config")

        self.config = Config.load_config(self.config_path)
        self.model_config = ModelConfig(**self.config.pop("model"))
        self.dataset_config = DatasetConfig(**self.config.pop("dataset"))
        self.training_config = TrainingArgumentsConfigImporter(
            **self.config.pop("training_args")
        ).get_config()
        self.lora_config = LoRAConfigImporter(**self.config.pop("lora")).get_config()
        self.bitsandbytes_config = BitsAndBytesConfigImporter(
            **self.config.pop("bitsandbytes")
        ).get_config()
        self.sft_config = SFTTConfig(**self.config.pop("sftt"))

    def load_dataset(self):
        print_rich("Load Dataset")
        self.dataset = load_dataset(
            self.dataset_config.name, split=self.dataset_config.split
        )
        print_rich(f"Len of Dataset {len(self.dataset)}")  # type: ignore

    def init_model(self):
        print_rich("Init Model")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.base_model,
            quantization_config=self.bitsandbytes_config,
            attn_implementation="flash_attention_2",
            use_cache=self.model_config.use_cache,
            device_map="auto",
        )

    def init_tokenizer(self):
        print_rich("Init Tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.base_model)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.padding_side = "right"
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def init_trainer(self):
        print_rich("Init Trainer")
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset,  # type: ignore
            peft_config=self.lora_config,  # type: ignore
            dataset_text_field="prompt",
            tokenizer=self.tokenizer,
            max_seq_length=self.sft_config.max_seq_length,
            packing=self.sft_config.packing,
            args=self.training_config,
        )

    def train(self):
        print_rich("Start Training")
        self.trainer.train()  # type: ignore
        print_rich("Finish Training")

    def save_model(self):
        print_rich("Start Saving Model")
        self.trainer.model.save_pretrained(f"shared/models/{self.model_config.new_model}", save_embedding_layers=True, save_adapter=True, save_config=True)  # type: ignore
        print_rich("Finish Saving Model")

    def save_run_results(self):
        config_dict = {}
        for config in self.config:
            for key, value in self.config[config].items():
                config_dict[key] = value
        return config_dict


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, help="give the run id", required=True)
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.wandb:
        wandb.init(project="ma-llm-train", name=f"{args.run_id}_trainer")
        wandb.config.update(args)

    trainer = Trainer(args.config_path, args.wandb)
    trainer.load_configs()
    trainer.load_dataset()
    trainer.init_model()
    trainer.init_tokenizer()
    trainer.init_trainer()
    trainer.train()
    trainer.save_model()

    if args.wandb:
        wandb.log(trainer.save_run_results())
        wandb.finish()
