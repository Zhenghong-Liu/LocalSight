"""训练基础设施：Muon/AdamW、梯度裁剪、checkpoint、各阶段训练循环。"""

from .muon import Muon, zeropower_via_newtonschulz5

__all__ = ["Muon", "zeropower_via_newtonschulz5"]
