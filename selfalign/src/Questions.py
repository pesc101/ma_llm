import random
from enum import Enum


class QuestionType(Enum):
    SYNTAX = "syntax"
    DEPENDENCY = "dependency"
    META = "meta"


class QuestionFilePaths(Enum):
    root_folder = ""
    SYNTAX = f"{root_folder}/ma_llm/selfalign/data/questions/syntax.txt"
    DEPENDENCY = f"{root_folder}/ma_llm/selfalign/data/questions/dependency.txt"
    META = f"{root_folder}/ma_llm/selfalign/data/questions/meta.txt"


def get_file_path(question_type: QuestionType):
    if question_type == QuestionType.SYNTAX:
        return QuestionFilePaths.SYNTAX.value
    elif question_type == QuestionType.DEPENDENCY:
        return QuestionFilePaths.DEPENDENCY.value
    elif question_type == QuestionType.META:
        return QuestionFilePaths.META.value
    else:
        return "unknown"


class Question:
    def __init__(self):
        self.question_type = random.choice(list(QuestionType))
        self.question_data = self.parse_question()

    def parse_question(self):
        file_path = get_file_path(self.question_type)
        return self.get_question(file_path, self.question_type)

    def get_question(self, file_path: str, question_type: QuestionType):
        with open(file_path, "r") as file:
            questions = file.readlines()
        return {"type": f"{question_type}", "content": random.choice(questions)}
