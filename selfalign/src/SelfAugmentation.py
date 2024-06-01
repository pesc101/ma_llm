import argparse

import pretty_errors
import wandb

from selfalign.src.custom_datasets.DatasetFactory import DatasetFactory
from shared.src.InferenceLLM import InferenceLLM
from shared.src.utils.io import dump_jsonlines, load_jsonlines


class SelfAugmentation:
    def __init__(
        self,
        inference_llm: InferenceLLM,
        model_name: str,
        dataset_type: str,
        unlabelled_data_filepath: str,
        dataset_factor: int = 1,
    ):
        self.inference_llm = inference_llm
        self.dataset_factor = dataset_factor
        self.raw_data = load_jsonlines(unlabelled_data_filepath)
        self.dataset = DatasetFactory.get_dataset_class(
            self.raw_data,
            model_name,
            dataset_type,
        )
        prompts = []
        for i in range(dataset_factor):
            prompts += self.dataset.get_all_prompts()
            print(f"{i*len(self.raw_data)} / {dataset_factor*len(self.raw_data)} generated prompts.")
        self.prompts = prompts

    def generate(self):
        self.results = self.inference_llm.generate(self.prompts)

    def save_results(self, output_filepath: str):
        if self.dataset is not None:
            results_to_dump = self.dataset.format_output(self.results, self.dataset_factor)
            dump_jsonlines(results_to_dump, output_filepath)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlabelled_data_filepath", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--model_name", type=str, default="mistral")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--dataset_factor", type=int, default=1)
    parser.add_argument("--dataset_type", type=str)
    parser.add_argument("--output_filepath", type=str)
    parser.add_argument("--gpu_memory_utilization", type=float)
    parser.add_argument("--run_id", type=str, help="give the run id", required=True)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.wandb:
        wandb.init(project="ma-llm-self-aug", name=f"{args.run_id}_self_aug")
        wandb.config.update(args)


    inference_llm = InferenceLLM(
        args.temperature,
        args.top_p,
        args.max_new_tokens,
        args.model_path,
        args.tensor_parallel_size,
        args.dtype,
        args.gpu_memory_utilization,
    )
    augmentation = SelfAugmentation(
        inference_llm, args.model_name, args.dataset_type, args.unlabelled_data_filepath, args.dataset_factor
    )
    augmentation.generate()
    augmentation.save_results(args.output_filepath)
    
    if args.wandb:
        wandb.finish()

