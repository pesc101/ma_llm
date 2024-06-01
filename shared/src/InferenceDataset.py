from abc import abstractmethod
from fastchat.conversation import Conversation, get_conv_template
from torch.utils.data import Dataset

from shared.src.ChatTemplates import ChatTemplates


class InferenceDataset(Dataset):
    def __init__(self, data: list, template_name: str, model_name: str = "mistral"):
        ChatTemplates()
        self.data = data
        self.conv: Conversation = get_conv_template(f'{model_name}-{template_name}')
        self.model_name = model_name

    def __len__(self):
        return len(self.data)

    def get_all_prompts(self):
        return [self[i] for i in range(len(self))]

    def __getitem__(self, idx):
        ins = self.data[idx]
        self.conv.messages.clear()
        self.conv.append_message(self.conv.roles[0], self.create_message(ins))  # type: ignore
        self.conv.append_message(self.conv.roles[1], None)  # type: ignore
        prompt = self.conv.get_prompt()
        return prompt

    @abstractmethod
    def create_message(self, ins):
        pass

    def format_output(self, results):
        return results
