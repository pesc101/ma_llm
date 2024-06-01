# %%
import argparse
import json
import os
import random
import re

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from shared.src.utils.io import read_yml


def ask_assistant(client, sys_prompt: str, user_prompt: str) -> str:
    # create a message
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=1.5,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].message.content


def create_user_input(instruction: str, df: pd.DataFrame) -> str:
    return f"{instruction} \n {df.to_string(index=False, header=True)}"


def generate_outputs(
    df_ast_imports, target_files, client, sys_prompt, instruction, factor=1, subsample=5
):
    output_list = []
    file_list = []
    for file in tqdm(target_files[:subsample], total=len(target_files[:subsample])):
        for i in range(factor):
            filtered_df = df_ast_imports[df_ast_imports["target_file"] == file]
            user_prompt = create_user_input(instruction, filtered_df)
            output_list.append(ask_assistant(client, sys_prompt, user_prompt))
            file_list.append(file)
    return output_list, file_list


def reformat_output(output: str, target_file: str) -> pd.DataFrame:
    if "Question:" not in output or "Answer:" not in output:
        return pd.DataFrame(columns=["question", "answer", "target_file", "raw_output"])

    split_output = re.split(r"\s*Question:\s*", output)[1:]
    question_answer_pairs = []

    for segment in split_output:
        if "Answer:" in segment:
            question_str = re.split(r"\s*Answer:\s*", segment)[0].strip()
            answer_str = re.split(r"\s*Answer:\s*", segment)[1].strip()
            question_answer_pairs.append(
                (question_str, answer_str, target_file, output)
            )
        else:
            return pd.DataFrame(
                columns=["question", "answer", "target_file", "raw_output"]
            )

    return pd.DataFrame(
        question_answer_pairs,
        columns=["question", "answer", "target_file", "raw_output"],
    )


def generate_question_answer_pairs(output_list, file_list):
    pairs = list(map(reformat_output, output_list, file_list))
    return pd.concat(pairs)


def write_output_to_file(df, output_file_path):
    reformatted_json = df.to_dict(orient="records")
    with open(output_file_path, "w") as output_file:
        for record in reformatted_json:
            output_file.write(json.dumps(record) + "\n")


def main():
    root_folder = ""
    parser = argparse.ArgumentParser(description="Generate question-answer pairs.")
    parser.add_argument("--api_key", type=str, help="API key for OpenAI")
    parser.add_argument(
        "--sys_prompt_path",
        type=str,
        help="Path to system prompt file",
        default=f"{root_folder}/ma_llm/evaluation/prompts/dependencies_generator.txt",
    )
    parser.add_argument(
        "--dependencies_data_file_path",
        type=str,
        help="Path to DataFrame file",
        default=f"{root_folder}/ma_llm/evaluation/results/spyder.xlsx",
    )
    parser.add_argument(
        "--output_file_path",
        type=str,
        help="Path to output file",
        default=f"{root_folder}/ma_llm/evaluation/results/spyder_question_answer_pairs.jsonl",
    )
    parser.add_argument("--sample_size", type=int, help="Subsample the data", default=5)

    args = parser.parse_args()

    secrets_file_path = f"{root_folder}/ma_llm/secrets.yml"
    if os.path.exists(secrets_file_path):
        api_key = read_yml(secrets_file_path)["openai"]["api_key"]
    else:
        api_key = args.api_key

    client = OpenAI(api_key=api_key)
    with open(args.sys_prompt_path, "r") as f:
        sys_prompt = f.read()

    df_ast_imports = pd.read_excel(args.dependencies_data_file_path)

    instruction = "Create me a question answer pair from the following table:"

    target_files = df_ast_imports["target_file"].unique()
    random.shuffle(target_files)

    output_list, file_list = generate_outputs(
        df_ast_imports,
        target_files,
        client,
        sys_prompt,
        instruction,
        1,
        args.sample_size,
    )
    df = generate_question_answer_pairs(output_list, file_list)

    write_output_to_file(df, args.output_file_path)


if __name__ == "__main__":
    main()
