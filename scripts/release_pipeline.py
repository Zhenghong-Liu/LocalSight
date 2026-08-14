#!/usr/bin/env python3
"""M9 发布管线：GGUF 转换 → Q8_0/Q4_K_M 量化 → Ollama（用户态）→ 冒烟验证。"""
from __future__ import annotations

import datetime
import os
import shutil
import subprocess
import urllib.request
import tarfile
import io
from pathlib import Path

B = Path("/home/sodastar/project/LocalSight")
REL = B / "artifacts" / "release"
LOG = B / "artifacts" / "release.log"
ENV = {**os.environ, "PYTHONPATH": f"{B}/src"}


def log(msg: str) -> None:
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now():%H:%M:%S}] {msg}\n")


def run(cmd: list[str], name: str, check: bool = True) -> int:
    log(f"=== {name}: {' '.join(cmd[:5])}... ===")
    result = subprocess.run(cmd, cwd=B, env=ENV, start_new_session=True)
    log(f"=== {name} exit={result.returncode} ===")
    if check and result.returncode != 0:
        raise SystemExit(f"{name} 失败")
    return result.returncode


def install_ollama() -> Path:
    home = Path.home() / "ollama"
    if (home / "bin/ollama").exists():
        return home / "bin/ollama"
    log("下载 Ollama（用户态）")
    data = urllib.request.urlopen(
        "https://ollama.com/download/ollama-linux-amd64.tgz", timeout=600
    ).read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        tf.extractall(Path.home())
    return home / "bin/ollama"


def main() -> None:
    REL.mkdir(parents=True, exist_ok=True)
    run(
        [str(B / ".venv/bin/python"), str(B / "scripts/convert_to_gguf.py"),
         "--checkpoint", str(B / "artifacts/agent_rl"),
         "--out", str(REL / "localsight.f16.gguf"),
         "--tokenizer", str(B / "data/tokenizer")],
        "GGUF 转换",
    )
    quantize = str(B / "tools/llama.cpp/build/bin/llama-quantize")
    for name, qtype in (("Q8_0", "q8_0"), ("Q4_K_M", "q4_k_m")):
        run([quantize, str(REL / "localsight.f16.gguf"),
             str(REL / f"localsight.{name}.gguf"), qtype], f"量化 {name}")

    ollama = install_ollama()
    env = {**ENV, "OLLAMA_MODELS": str(B / "models/ollama")}
    subprocess.Popen(
        [str(ollama), "serve"], cwd=B, env=env, start_new_session=True,
        stdout=open(B / "artifacts/ollama_serve.log", "w"),
        stderr=subprocess.STDOUT,
    )
    log("ollama serve 启动")
    import time
    time.sleep(15)

    modelfile = REL / "Modelfile"
    template = (B / "deploy/ollama/Modelfile.template").read_text(encoding="utf-8")
    modelfile.write_text(
        template.replace("./localsight-198m.Q4_K_M.gguf", str(REL / "localsight.Q4_K_M.gguf")),
        encoding="utf-8",
    )
    run([str(ollama), "create", "localsight-198m", "-f", str(modelfile)], "ollama create", check=False)
    run([str(ollama), "run", "localsight-198m", "你好，请用一句话介绍你自己。"], "ollama 冒烟", check=False)
    log("发布管线完成")


if __name__ == "__main__":
    main()
