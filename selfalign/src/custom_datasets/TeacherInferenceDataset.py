from os import error

from shared.src.InferenceDataset import InferenceDataset
from selfalign.src.Questions import Question


class TeacherInferenceDataset(InferenceDataset):
    def __init__(self, data: list, model_name: str = "mistral"):
        template_name = "teacher"
        super().__init__(data, template_name, model_name)

    
    # TODO FIX BUG: Question content differs from the question data
    def create_message(self, ins) -> str:

        question = Question()
        ins["question"] = question.question_data
        question_content = question.question_data["content"]
        metadata = self.__reformat_meta_data(ins["meta_data"])

        if self.model_name == "mistral":
            return (
                f"\nQuestion: {question_content}\n{metadata}<</SYS>>\n\n{ins['code']}"
            )
        else:
            error(f"Model name {self.model_name} not supported")
        return ""

    def __reformat_meta_data(self, meta_data: dict) -> str:
        return "Meta Data: \n" + "".join(
            [f"##{k}: {v}\n" for k, v in meta_data.items()]
        )

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
                        "teacher_full_prompt": response.prompt,
                        "teacher_response": response.outputs[0].text,
                    },
                }
            )
        return results_to_dump
