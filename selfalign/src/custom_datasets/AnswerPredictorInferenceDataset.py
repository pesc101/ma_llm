from shared.src.InferenceDataset import InferenceDataset


class AnswerPredictorInferenceDataset(InferenceDataset):
    def __init__(self, data: list, model_name: str = "mistral"):
        template_name = "answer-predictor"
        super().__init__(data, template_name, model_name)

    def create_message(self, ins):
        return ins["conversation"]["teacher_response"]

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
                        "answer_pred_full_prompt": response.prompt,
                        "answer_pred_response": response.outputs[0].text,
                    },
                }
            )
        return results_to_dump
