# %%
import argparse
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pretty_errors
import wandb

root_folder = ""
sys.path.append(root_folder)
from tqdm import tqdm

from shared.src.OpenAIClient import OpenAIClient
from shared.src.utils.io import dump_json, dump_jsonlines, load_jsonlines, read_yml


class ModelBasedEvaluator:
    def __init__(
        self,
        run_id: str,
        judge: OpenAIClient,
        qa_dataset: list[dict],
        answers1: list[dict],
        answers2: list[dict],
        sys_prompt_path: str,
    ):
        self.run_id = run_id
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.judge = judge
        self.qa_dataset = qa_dataset
        self.answers1 = answers1
        self.answers2 = answers2
        self.prompts = []
        self.results = []
        self.__build_sys_prompt(sys_prompt_path)
        self.__build_df()

    def __build_sys_prompt(self, sys_prompt_path: str):
        with open(sys_prompt_path, "r") as file:
            self.sys_prompt = file.read()

    def __build_df(self):
        df = pd.json_normalize(self.qa_dataset)
        df_answer1 = pd.json_normalize(self.answers1)
        df_answer1 = df_answer1.rename(columns={"response": "model_answer1"})
        df_answer2 = pd.json_normalize(self.answers2)
        df_answer2 = df_answer2.rename(columns={"response": "model_answer2"})
        self.df = pd.concat(
            [
                df,
                df_answer1["model_answer1"],
                df_answer2["model_answer2"],
            ],
            axis=1,
        )

        model_name_1 = df_answer1["model"][0]
        if df_answer1["rag"][0]:
            model_name_1 += "_rag"
        self.df["model_name1"] = model_name_1

        model_name_2 = df_answer2["model"][0]
        if df_answer2["rag"][0]:
            model_name_2 += "_rag"
        self.df["model_name2"] = model_name_2
        self.df["prompt"] = ""

    def evaluate(self):
        self.__build_prompts()
        responses = []
        for _, row in tqdm(
            self.df.iterrows(), desc="Evaluating", total=self.df.shape[0]
        ):
            _, response = self.judge.generate(self.sys_prompt, row["prompt"])
            responses.append(response)
        self.df["response"] = responses
        self.__determine_results()

    def __build_prompts(self):
        self.df["split"] = np.random.choice([0, 1], size=len(self.df))
        for index, row in self.df.iterrows():
            if row["split"] == 0:
                answer_A, answer_B = (
                    row["model_answer1"],
                    row["model_answer2"],
                )
            elif row["split"] == 1:
                answer_A, answer_B = (
                    row["model_answer2"],
                    row["model_answer1"],
                )
            else:
                raise ValueError("Split must be 0 or 1")
            self.df.at[index, "prompt"] = self.__build_prompt(
                row, str(answer_A), str(answer_B)
            )

    def __build_prompt(self, row, answer1: str, answer2: str):
        user_prompt = f'[User Question] {row["question"]} [End of User Question] [Model Solution] {row["answer"]} [End of Model Solution] [The Start of Assistant A’s Answer] {answer1} [The End of Assistant A’s Answer] [The Start of Assistant B’s Answer] {answer2} [The End of Assistant B’s Answer]'
        return user_prompt

    def __determine_results(self):
        self.df["extracted_winner_str"] = self.df["response"].apply(
            self.__extract_field_from_line
        )
        winner_list = []
        for index, row in self.df.iterrows():
            if row["split"] == 0 and row["extracted_winner_str"] == "A":
                winner_list.append(row["model_name1"])
            elif row["split"] == 0 and row["extracted_winner_str"] == "B":
                winner_list.append(row["model_name2"])
            elif row["split"] == 1 and row["extracted_winner_str"] == "A":
                winner_list.append(row["model_name2"])
            elif row["split"] == 1 and row["extracted_winner_str"] == "B":
                winner_list.append(row["model_name1"])
            elif row["extracted_winner_str"] == "C":
                winner_list.append("Tie Good")
            elif row["extracted_winner_str"] == "D":
                winner_list.append("Tie Bad")
            else:
                winner_list.append("No Value")
        self.df["winner"] = winner_list
        print("Results determined")

    def __extract_field_from_line(self, response: str):
        match = re.search(r"\[\[(A|B|C|D)\]\]", response)
        return match.group(1) if match else None

    def __format_output(self):

        results_to_dump = []
        for _, row in self.df.iterrows():
            results_to_dump.append(
                {
                    "run_id": self.run_id,
                    "question": row["question"],
                    "type": row["type"],
                    "model_name1": row["model_name1"],
                    "model_name2": row["model_name2"],
                    "full_prompt": row["prompt"],
                    "response": row["response"],
                    "winner": row["winner"],
                },
            )
        return results_to_dump

    def format_run_results(self):
        return [
            {
                "run_id": self.run_id,
                "timestamp": self.timestamp,
                "judge_name": self.judge.model,
                "model_parameter": {
                    "temperature": self.judge.temperature,
                    "max_tokens": self.judge.max_tokens,
                    "top_p": self.judge.top_p,
                    "frequency_penalty": self.judge.frequency_penalty,
                    "presence_penalty": self.judge.presence_penalty,
                },
                "model_name1": self.df["model_name1"].iloc[0],
                "model_name2": self.df["model_name2"].iloc[0],
                "sys_prompt": self.sys_prompt,
                "results": self.rel_freq,
            }
        ]

    def calculate_scores(self):
        df = pd.DataFrame({"type": self.df["type"], "winner": self.df.winner})
        df_grouped = (
            df.groupby("type")["winner"]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
            .T
        )
        self.rel_freq = df_grouped.to_dict()
        print("Scores calculated: ", self.rel_freq)

    def save_output_to_json(self, output_filepath: str):
        dump_jsonlines(self.__format_output(), output_filepath)

    def save_run_results(self, output_filepath: str):
        dump_json(self.format_run_results(), output_filepath)

    def save_df(self, output_filepath: str):
        self.df.to_csv(output_filepath)


# %%
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--output_filepath", type=str)
    parser.add_argument("--sys_prompt_path", type=str)
    parser.add_argument("--qa_data_path", type=str)
    parser.add_argument("--answers1_path", type=str)
    parser.add_argument("--answers2_path", type=str)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction)
    parser.add_argument("--dataset_name", type=str, default="all")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.wandb:
        wandb.init(project="ma-llm", name=f"{args.run_id}_model_eval")
        wandb.config.update(args)

    root_folder = ""
    secrets_file_path = f"{root_folder}/ma_llm/secrets.yml"
    if os.path.exists(secrets_file_path):
        api_key = read_yml(secrets_file_path)["openai"]["api_key"]
    else:
        api_key = args.api_key

    client = OpenAIClient(
        api_key=api_key,
        model=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        frequency_penalty=0,
        presence_penalty=0,
    )
    with open(args.sys_prompt_path, "r") as f:
        sys_prompt = f.read()

    qa_dataset = load_jsonlines(args.qa_data_path)
    if args.dataset_name != "all":
        qa_dataset = [qa for qa in qa_dataset if qa.get("type") == args.dataset_name]

    answers1 = load_jsonlines(args.answers1_path)
    answers2 = load_jsonlines(args.answers2_path)

    evaluator = ModelBasedEvaluator(
        run_id=args.run_id,
        judge=client,
        qa_dataset=qa_dataset,
        answers1=answers1,
        answers2=answers2,
        sys_prompt_path=args.sys_prompt_path,
    )
    evaluator.evaluate()
    evaluator.calculate_scores()
    evaluator.save_output_to_json(args.output_filepath + "_raw.json")
    evaluator.save_df(args.output_filepath + "_df.csv")
    if args.wandb:
        wandb.log(evaluator.format_run_results()[0])
        wandb.finish()
