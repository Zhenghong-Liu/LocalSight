"""基于 vLLM 的本地 LLM judge。

模型下载：服务器上设 HF_ENDPOINT=https://hf-mirror.com（国内镜像）再首次加载。
"""
from __future__ import annotations

from .judge import JudgeClient, JudgeResult, build_judge_prompt, parse_judge_score


class VLLMJudge(JudgeClient):
    def __init__(self, model: str, tensor_parallel: int = 1, max_model_len: int = 8192):
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel,
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=0.95,
        )
        self.sampling = SamplingParams(temperature=0.0, max_tokens=256)

    def score(self, question: str, answer: str) -> JudgeResult:
        prompt = build_judge_prompt(question, answer)
        output = self.llm.generate([prompt], self.sampling)
        raw = output[0].outputs[0].text
        score = parse_judge_score(raw)
        return JudgeResult(score if score is not None else 0.0, raw)
