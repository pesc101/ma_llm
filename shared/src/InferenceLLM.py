from typing import List

from vllm import LLM, SamplingParams


class InferenceLLM:
    def __init__(
        self,
        temperature: float,
        top_p: float,
        max_tokens: int,
        model: str,
        tensor_parallel_size: int,
        dtype: str,
        gpu_memory_utilization: float,
    ):

        self.llm = LLM(
            model,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            max_model_len=max_tokens,
            gpu_memory_utilization= gpu_memory_utilization
        )
        self.sampling_parameters = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.model = model
        self.tensor_parallel_size = tensor_parallel_size

    def generate(self, prompts: List[str]):
        return self.llm.generate(
            prompts, use_tqdm=True, sampling_params=self.sampling_parameters
        )
