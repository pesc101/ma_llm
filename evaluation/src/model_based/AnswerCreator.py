# %%
import argparse
import os
import sys
from abc import abstractmethod

import pretty_errors
import wandb
from tqdm import tqdm

root_folder = ""
sys.path.append(root_folder)

from rag.src.client import RAGClient
from shared.src.InferenceLLM import InferenceLLM
from shared.src.OpenAIClient import OpenAIClient
from shared.src.utils.io import dump_json, dump_jsonlines, load_jsonlines, read_yml
from shared.src.utils.print import print_rich


class AnswerCreatorFactory:
    @staticmethod
    def create(
        model: InferenceLLM | OpenAIClient,
        qa_dataset: list[dict],
        sys_prompt_path: str,
        rag_flag: int = 0,
        rag_collection: str = "spyder-embeddings",
        rag_n=1,
        meta_data_flag: int = 0,
    ):
        if rag_flag == 1:
            rag_client = RAGClient(
                embeddings_dir=f"{root_folder}/ma_llm/rag/embeddings"
            )
            return AnswerCreatorRAG(model, rag_client, rag_collection, qa_dataset, sys_prompt_path, meta_data_flag=meta_data_flag, n=rag_n)  # type: ignore
        elif isinstance(model, InferenceLLM):
            return AnswerCreatorLocal(model, qa_dataset, sys_prompt_path)
        elif isinstance(model, OpenAIClient):
            return AnswerCreatorOpenAI(model, qa_dataset, sys_prompt_path)
        else:
            raise ValueError("Model type not supported")


class AnswerCreator:
    def __init__(
        self,
        model: InferenceLLM | OpenAIClient,
        qa_dataset: list[dict],
        sys_prompt_path: str,
    ):
        self.qa_dataset = qa_dataset
        self.model = model
        self.answers = []
        self.prompts = []
        self.full_prompts = []
        self.__build_sys_prompt(sys_prompt_path)

    def __build_sys_prompt(self, sys_prompt_path: str):
        with open(sys_prompt_path, "r") as file:
            self.sys_prompt = file.read()

    def save_output_to_json(self, output_filepath: str):
        dump_jsonlines(self.format_output(), output_filepath)

    def save_run_results(self, output_filepath: str):
        dump_json(self.format_run_results(), output_filepath)

    @abstractmethod
    def format_output(self):
        pass

    @abstractmethod
    def format_run_results(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass


class AnswerCreatorLocal(AnswerCreator):
    def __init__(
        self, model: InferenceLLM, qa_dataset: list[dict], sys_prompt_path: str
    ):
        super().__init__(model, qa_dataset, sys_prompt_path)

    def __build_prompt(self, qa: dict):
        return f'[INST] <<SYS>> {self.sys_prompt} <</SYS>> \n\n<<[User Question] {qa.get("question")} [End of User Question] [/INST]'

    def __build_prompts(self):
        print_rich("Building Prompts...")
        self.prompts = list(map(self.__build_prompt, self.qa_dataset))  # type: ignore
        print_rich("... Prompts Built")

    def evaluate(self):
        self.__build_prompts()
        response = self.model.generate(self.prompts)  # type: ignore
        for i in range(len(response)):
            self.answers.append(response[i].outputs[0].text)
            self.full_prompts.append(response[i].prompt)

    def format_run_results(self):
        return [
            {
                "model_name": self.model.model,
                "model_parameter": {
                    "temperature": self.model.temperature,
                    "max_tokens": self.model.max_tokens,
                    "top_p": self.model.top_p,
                },
                "dataset_length": len(self.qa_dataset),
                "sys_prompt": self.sys_prompt,
            }
        ]

    def format_output(self):
        results_to_dump = []
        for prompt, answer in zip(self.full_prompts, self.answers):
            results_to_dump.append(
                {
                    "model": self.model.model,
                    "prompt": prompt,
                    "response": answer,
                    "rag": False,
                },
            )
        return results_to_dump


class AnswerCreatorOpenAI(AnswerCreator):
    def __init__(
        self, model: OpenAIClient, qa_dataset: list[dict], sys_prompt_path: str
    ):
        super().__init__(model, qa_dataset, sys_prompt_path)

    def __build_prompt(self, qa: dict):
        return f'[User Question] {qa.get("question")} [End of User Question]'

    def __build_prompts(self):
        self.prompts = list(map(self.__build_prompt, self.qa_dataset))  # type: ignore

    def evaluate(self):
        self.__build_prompts()
        for prompt in tqdm(self.prompts, desc="Creating Answers"):
            full_prompt, answer = self.model.generate(self.sys_prompt, prompt)  # type: ignore
            self.answers.append(answer)
            self.full_prompts.append(full_prompt)

    def format_run_results(self):
        return [
            {
                "model_name": self.model.model,
                "model_parameter": {
                    "temperature": self.model.temperature,
                    "max_tokens": self.model.max_tokens,
                    "top_p": self.model.top_p,
                    "frequency_penalty": self.model.frequency_penalty,  # type: ignore
                    "presence_penalty": self.model.presence_penalty,  # type: ignore
                },
                "dataset_length": len(self.qa_dataset),
                "sys_prompt": self.sys_prompt,
            }
        ]

    def format_output(self):
        results_to_dump = []
        for prompt, answer in zip(self.full_prompts, self.answers):
            results_to_dump.append(
                {
                    "model": self.model.model,
                    "prompt": prompt,
                    "response": answer,
                    "rag": False,
                },
            )
        return results_to_dump


class AnswerCreatorRAG(AnswerCreator):
    def __init__(
        self,
        model: InferenceLLM,
        rag_client: RAGClient,
        rag_collection: str,
        qa_dataset: list[dict],
        sys_prompt_path: str,
        rag_keys: list = ["file_name", "module"],
        n: int = 1,
        meta_data_flag: int = 0,
    ):
        self.rag_client = rag_client
        self.rag_collection = rag_collection
        self.n = n
        self.rag_keys = rag_keys
        self.meta_data_flag = meta_data_flag
        super().__init__(model, qa_dataset, sys_prompt_path)

    def evaluate(self):
        print_rich("Using RAG model with Collection: " + self.rag_collection + "...")
        self.__build_prompts()
        response = self.model.generate(self.prompts)  # type: ignore
        for i in range(len(response)):
            self.answers.append(response[i].outputs[0].text)
            self.full_prompts.append(response[i].prompt)

    def __build_prompt(self, qa: dict):
        meta_data = {}
        if self.meta_data_flag == 1:
            meta_data = qa.get("meta_data")
            meta_data = [{key: meta_data[key]} for key in self.rag_keys]  # type: ignore
            meta_data = {"$and": meta_data}

        # Get documents
        documents = self.rag_client.query(
            collection_name=self.rag_collection,
            query=qa["question"],
            meta_data=meta_data,
            n=self.n,
        )
        documents = [item for sublist in documents["documents"] for item in sublist]
        joined_documents = ("\n Snippet: ").join(documents)

        # Build Prompt
        prompt = f"[INST] <<SYS>> {self.sys_prompt} <</SYS>> \n\n"

        # Test if documents where found
        if len(documents) > 0:
            prompt += f"Answer the question using the provided context. \n"
            prompt += f"\\n\\n Context: {joined_documents} \n\n"
        prompt += f'Question: {qa["question"]} \\n\\n Answer:",\n'
        prompt += "[/INST]"

        return prompt

    def __build_prompts(self):
        self.prompts = list(map(self.__build_prompt, self.qa_dataset))  # type: ignore

    def format_run_results(self):
        return [
            {
                "model_name": self.model.model,
                "model_parameter": {
                    "temperature": self.model.temperature,
                    "max_tokens": self.model.max_tokens,
                    "top_p": self.model.top_p,
                },
                "dataset_length": len(self.qa_dataset),
                "sys_prompt": self.sys_prompt,
            }
        ]

    def format_output(self):
        results_to_dump = []
        for prompt, answer in zip(self.full_prompts, self.answers):
            results_to_dump.append(
                {
                    "model": self.model.model,
                    "prompt": prompt,
                    "response": answer,
                    "rag": True,
                },
            )
        return results_to_dump


# %%
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--output_filepath", type=str)
    parser.add_argument("--sys_prompt_path", type=str)
    parser.add_argument("--qa_data_path", type=str)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--rag_flag", type=int, default=0)
    parser.add_argument("--rag_collection", type=str, default="spyder-embeddings")
    parser.add_argument("--rag_n", type=int)
    parser.add_argument("--meta_data_flag", type=int, default=0)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction)
    parser.add_argument("--dataset_name", type=str, default="all")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.wandb:
        wandb.init(project="ma-llm", name=f"{args.run_id}_answer_creator")
        wandb.config.update(args)

    ## if substring gpt3 is in model_name then create OPENAI client
    root_folder = ""
    secrets_file_path = f"{root_folder}/ma_llm/secrets.yml"
    if "gpt" in args.model_name:
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
    else:
        ## else create InferenceLLM client
        client = InferenceLLM(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            model=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

    with open(args.sys_prompt_path, "r") as f:
        sys_prompt = f.read()

    qa_dataset = load_jsonlines(args.qa_data_path)
    if args.dataset_name != "all":
        qa_dataset = [qa for qa in qa_dataset if qa.get("type") == args.dataset_name]

    answer_creator = AnswerCreatorFactory.create(
        model=client,
        qa_dataset=qa_dataset,
        sys_prompt_path=args.sys_prompt_path,
        rag_flag=args.rag_flag,
        rag_collection=args.rag_collection,
        rag_n=args.rag_n,
        meta_data_flag=args.meta_data_flag,
    )
    answer_creator.evaluate()
    answer_creator.save_output_to_json(args.output_filepath)

    if args.wandb:
        wandb.log(answer_creator.format_run_results()[0])
        wandb.finish()
