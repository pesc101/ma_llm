# %%
import argparse
import re
import sys

from datasets import Dataset

root_folder = ""
sys.path.append(root_folder)

from selfalign.src.Questions import QuestionType, get_file_path
from shared.src.utils.io import load_jsonlines


class DatasetTransformer:
    def __init__(self, dataset: list, output_dataset_name: str):
        self.dataset = Dataset.from_list(dataset)
        self.output_dataset_name = output_dataset_name

    def create_dataset(self):
        all_prompts = []
        for sample in self.dataset:
            prompts = self.transform_sample(sample)
            all_prompts.extend(prompts)
        self.reformatted_dataset = Dataset.from_list(all_prompts)

    def transform_sample(self, sample: dict) -> list:
        sample["conversation"]["teacher_question"] = self.__get__teacher_question(
            sample["conversation"]["teacher_full_prompt"]
        )
        sample["conversation"]["question_type"] = self.__verify_question_type(
            sample["conversation"]["teacher_question"]
        )
        raw_qa_sample = sample["conversation"]["answer_pred_response"]
        questions, answers = extract_questions_and_answers(raw_qa_sample)

        samples = []
        for question, answer in zip(questions, answers):
            samples.append(
                {
                    "meta_data": sample["meta_data"],
                    "code": sample["code"],
                    "question": question,
                    "answer": answer,
                    "conversation": sample["conversation"],
                }
            )
        return samples

    def __verify_question_type(self, question: str):
        for question_type in QuestionType:
            file_path = get_file_path(question_type)
            try:
                with open(file_path, "r") as file:
                    for line in file:
                        if question == line.strip():
                            return str(question_type).split(".")[-1]
            except FileNotFoundError:
                print("File not found.")
        return "unknown"

    def __get__teacher_question(self, text: str):
        pattern = r"Question:(.*?)\n\n"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            extracted_text = match.group(1).strip()
            return extracted_text
        return ""

    def add_column(self):
        self.reformatted_dataset = self.reformatted_dataset.map(self.add_mistral_format)

    def add_mistral_format(self, sample: dict):
        return {
            "prompt": f"<s>[INST] {sample['question']} [/INST] {sample['answer']} </s>"
        }

    def push_to_hub(self):
        self.reformatted_dataset.push_to_hub(self.output_dataset_name)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selfalign_dataset_path", type=str)
    parser.add_argument("--output_dataset_name", type=str, default="test_dataset")
    return parser.parse_args()


def extract_questions_and_answers(text: str):
    question_pattern = r"Question:\s*(.*?)\s*Answer:\s*(.*?)\s*(?=\nQuestion|$)"
    matches = re.findall(question_pattern, text, re.DOTALL)

    questions, answers = [], []
    for match in matches:
        question = match[0].strip()
        answer = match[1].strip()
        questions.append(question)
        answers.append(answer)

    return questions, answers


# %%
if __name__ == "__main__":
    args = get_args()

    raw_data = load_jsonlines(args.selfalign_dataset_path)
    dataset = DatasetTransformer(raw_data, args.output_dataset_name)
    dataset.create_dataset()
    dataset.add_column()
    dataset.push_to_hub()
