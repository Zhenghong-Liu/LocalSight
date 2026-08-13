"""多模态扩展预留接口（当前不实现）。

规划：ModalityEncoder（视觉/音频编码器）+ Projector（特征→768 维），
主 Transformer 只认 hidden_states + position_ids。词表已含 vision/audio/TTS token。
"""
