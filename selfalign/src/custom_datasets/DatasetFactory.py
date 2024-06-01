# %%

from enum import Enum

from selfalign.src.custom_datasets.AnswerPredictorInferenceDataset import \
    AnswerPredictorInferenceDataset
from selfalign.src.custom_datasets.CurationEvaluationInferenceDataset import \
    CurationEvaluationInferenceDataset
from selfalign.src.custom_datasets.TeacherInferenceDataset import \
    TeacherInferenceDataset


class DatasetType(Enum):
    TEACHER = "teacher"
    ANSWER_PREDICTOR = "answer-predictor"
    CURATION = "curation"


class DatasetFactory:
    @classmethod
    def get_dataset_class(cls, dataset: list, model_name: str, dataset_type: str):
        if dataset_type == DatasetType.TEACHER.value:
            return TeacherInferenceDataset(dataset, model_name)
        elif dataset_type == DatasetType.ANSWER_PREDICTOR.value:
            return AnswerPredictorInferenceDataset(dataset, model_name)
        elif dataset_type == DatasetType.CURATION.value:
            return CurationEvaluationInferenceDataset(dataset, model_name)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
