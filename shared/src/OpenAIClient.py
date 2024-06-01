


from openai import OpenAI


class OpenAIClient:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo-1106", temperature: float = 1.5, max_tokens: int = 256, top_p: float = 1, frequency_penalty: float = 0, presence_penalty: float = 0):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    def generate(self, sys_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )
        answer = response.choices[0].message.content
        return messages, answer
