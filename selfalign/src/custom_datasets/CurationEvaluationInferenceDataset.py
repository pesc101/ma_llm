# %%
from os import error
import re

from shared.src.InferenceDataset import InferenceDataset


class CurationEvaluationInferenceDataset(InferenceDataset):
    def __init__(self, data: list, model_name: str = "mistral"):
        template_name = "curation"
        super().__init__(data, template_name, model_name)

    def create_message(self, ins):

        generarted_answer_response = ins["conversation"]["answer_pred_response"]
        generated_teacher_response = ins["conversation"]["teacher_response"]
        if self.model_name == "mistral":
            return f"\nInstruction: {generarted_answer_response}\n Answer: {generated_teacher_response}<</SYS>>\n\n"
        else:
            error(f"Model name {self.model_name} not supported")
        return ""

    def __get_score(self, curation_response: str) -> int | None:
        score_matched = re.compile(r"[Ss]core:\s*(\d+)").search(curation_response)
        if score_matched:
            return int(score_matched.group(1))
        else:
            score_matched2 = re.compile(r"as a \s*(\d+)").search(curation_response)
            return int(score_matched2.group(1)) if score_matched2 else None 

    def format_output(self, responses: list, dataset_factor: int = 1):
        temp_data = []
        for _ in range(dataset_factor):
            temp_data.extend(self.data)
        results_to_dump = []
        for raw, response in zip(temp_data, responses):
            results_to_dump.append(
                {
                    "meta_data": raw.get("meta_data", {}),
                    "code": raw.get("code", ""),
                    "question": raw.get("question", ""),
                    "conversation": {
                        "teacher_full_prompt": raw.get("conversation", "").get(
                            "teacher_full_prompt", ""
                        ),
                        "teacher_response": raw.get("conversation", "").get(
                            "teacher_response", ""
                        ),
                        "answer_pred_full_prompt": raw.get("conversation", "").get(
                            "answer_pred_full_prompt", ""
                        ),
                        "answer_pred_response": raw.get("conversation", "").get(
                            "answer_pred_response", ""
                        ),
                        "curation_full_prompt": response.prompt,
                        "curation_response": response.outputs[0].text,
                        "curation_score": self.__get_score(response.outputs[0].text),
                        "curation_length": len(response.outputs[0].text),
                    },
                }
            )
        return results_to_dump
