# %%
import argparse
import statistics as sts
from collections import Counter
from typing import Dict, List

import pretty_errors
import wandb

from shared.src.utils.io import dump_jsonlines, load_jsonlines
from shared.src.utils.print import print_rich


# %%
class SelfCuration:
    def __init__(
        self,
        data_filepath: str,
        save_filepath: str,
        min_inst_len: int,
        max_inst_len: int,
        threshold: int,
    ):
        self.data_filepath: str = data_filepath
        self.save_filepath: str = save_filepath
        self.min_inst_len: int = min_inst_len
        self.max_inst_len: int = max_inst_len
        self.threshold: int = threshold
        self.filtered_data: List[Dict] = []

    def create_df(self):
        data = load_jsonlines(self.data_filepath)
        self.data: List[Dict] = data

    def filter_curation(self):
        ## Filter Length
        self.filtered_data = [
            entry
            for entry in self.data
            if len(entry["conversation"]["curation_response"]) > self.min_inst_len
            and len(entry["conversation"]["curation_response"]) < self.max_inst_len
        ]
        ## Filter Score
        self.filtered_data = [
            entry
            for entry in self.filtered_data
            if entry["conversation"].get("curation_score") is not None
            and entry["conversation"]["curation_score"] >= self.threshold
        ]

    def create_stats(self):
        scores = [entry["conversation"]["curation_score"] for entry in self.data]
        score_overview = Counter(scores)
        distribution = [
            len(entry["conversation"]["curation_response"]) for entry in self.data
        ]
        length_distribution = Counter(distribution)
        return score_overview, length_distribution

    def print_statistics(self):
        score_overview, length_overview = self.create_stats()
        print_rich(f"Scores Distribution: {score_overview}")
        # print_rich(f"Length Distribution: {length_overview}")
        print_rich(
            f"Number of qualified results (Threshold: {self.threshold}, Min Length: {self.min_inst_len}, Max Length: {self.max_inst_len}): {len(self.filtered_data)}/{len(self.data)}"
        )
        response_lengths = [
            len(entry["conversation"]["curation_response"]) for entry in self.data
        ]
        mean_length = sts.mean(response_lengths)
        stdev_length = sts.stdev(response_lengths)
        print_rich(f"response len: {mean_length:.0f} ± {stdev_length:.0f}")

    def save_results(self):
        dump_jsonlines(self.filtered_data, self.save_filepath)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_filepath", type=str)
    parser.add_argument("--save_filepath", type=str)
    parser.add_argument("--min_inst_len", type=int, default=5)
    parser.add_argument("--max_inst_len", type=int, default=2000)
    parser.add_argument("--threshold", type=int, default=3)
    parser.add_argument("--run_id", type=str, help="give the run id", required=True)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.wandb:
        wandb.init(project="ma-llm-self-aug", name=f"{args.run_id}_self_curation")
        wandb.config.update(args)

    self_curation_filter = SelfCuration(
        args.data_filepath,
        args.save_filepath,
        args.min_inst_len,
        args.max_inst_len,
        args.threshold,
    )
    self_curation_filter.create_df()
    self_curation_filter.filter_curation()
    self_curation_filter.print_statistics()
    self_curation_filter.save_results()
    
    if args.wandb:
        wandb.finish()