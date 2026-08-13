"""LocalSight: 轻量、带思考能力的 198M-A64M MoE 语言模型。

规划阶段的包骨架。实现按 docs/ 中的规格推进：
- model/      自研 RMSNorm/RoPE/GQA/MoE/KV cache
- tokenizer/  复用数据目录提供的 6400 BPE 快照
- data/       清洗/tokenize/packing/mmap 加载
- training/   Muon/AdamW/各阶段损失与循环
- rl/         SimPO/GRPO/DAPO/采样/奖励/judge
- generation/ 推理采样
- eval/       评测
"""

__version__ = "0.1.0"
