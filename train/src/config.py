from dataclasses import dataclass
from typing import Optional

import torch
import yaml
from peft import LoraConfig
from transformers import BitsAndBytesConfig, TrainingArguments


class Config:
    @staticmethod
    def load_config(file_path: str):
        with open(file_path, "r") as file:
            return yaml.safe_load(file)


@dataclass
class LoRAConfigImporter:
    lora_r: int
    lora_alpha: int
    lora_dropout: float

    def get_config(self):
        return LoraConfig(
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            r=self.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
        )


@dataclass
class BitsAndBytesConfigImporter:
    use_4bit: bool
    bnb_4bit_compute_dtype: str
    bnb_4bit_quant_type: str
    use_nested_quant: bool

    def get_config(self):
        return BitsAndBytesConfig(
            load_in_4bit=self.use_4bit,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, self.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=self.use_nested_quant,
        )


@dataclass
class TrainingArgumentsConfigImporter:
    output_dir: str
    num_train_epochs: int
    fp16: bool
    bf16: bool
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    max_grad_norm: float
    learning_rate: float
    weight_decay: float
    optim: str
    lr_scheduler_type: str
    max_steps: int
    warmup_ratio: float
    group_by_length: bool
    save_steps: int
    logging_steps: int

    def get_config(self):
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            optim=self.optim,
            save_steps=self.save_steps,
            logging_steps=self.logging_steps,
            learning_rate=float(self.learning_rate),
            weight_decay=self.weight_decay,
            fp16=self.fp16,
            bf16=self.bf16,
            max_grad_norm=self.max_grad_norm,
            max_steps=self.max_steps,
            warmup_ratio=self.warmup_ratio,
            group_by_length=self.group_by_length,
            lr_scheduler_type=self.lr_scheduler_type,
            report_to="wandb",  # type: ignore
        )


@dataclass
class SFTTConfig:
    max_seq_length: Optional[int]
    packing: bool

@dataclass
class ModelConfig:
    base_model: str
    new_model: str
    device_map: str
    use_cache: bool = False
    pretraining_tp: int = 0
    tokenizer_padding: str = "rigth"


@dataclass
class DatasetConfig:
    name: str
    split: str
