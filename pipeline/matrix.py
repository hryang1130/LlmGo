#!/usr/bin/env python
"""
实验矩阵盘点: 把「模型线 × 阶段」的实验进度汇总成一张矩阵, 回答三个问题:
  做了什么 / 还缺什么 / 下一格怎么跑。

状态判定完全基于磁盘事实 (模型产物 + out/experiments/ 结构化记录), 不靠手工登记。

用法:
    python pipeline/matrix.py            # 生成 out/reports/matrix.md 并注册进 UI
    python pipeline/matrix.py --print    # 只打印, 不写文件
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import EXPERIMENTS_DIR, _register_report, markdown_table  # noqa: E402

DONE, TODO = "✅", "⬜"

# 实验矩阵定义: 每格 = (阶段, 完成判定 [产物路径/experiment 名], 怎么跑)
MATRIX = [
    {
        "line": "tiny-llama 线 (CPU · 3~4M 从零训练)",
        "cells": [
            ("预训练 train", ["models/tinyllm-f16.gguf", "models/tinyllm-hf"],
             "python pipeline/train.py --epochs 150"),
            ("指令微调 SFT", ["models/tinyllm-hf-sft", "experiment:sft"],
             "python pipeline/sft.py (--label tinyllm-sft)"),
            ("量化方案对比 (imatrix)", ["experiment:sweep"],
             "python pipeline/quantize_sweep.py --imatrix auto (--label tinyllm-sweep)"),
            ("速度 + PPL 评测", ["experiment:eval"],
             "python pipeline/eval.py --label tinyllm-eval"),
        ],
    },
    {
        "line": "qwen25-3b 线 (GPU · RTX 4060 Ti 8GB)",
        "cells": [
            ("QLoRA 微调", ["models/qwen25-3b-merged", "models/qwen25-3b-lora"],
             "python pipeline/finetune_qlora.py --samples 1200"),
            ("三档量化 F16/Q8_0/Q4_K_M",
             ["models/qwen25-3b-ft-f16.gguf", "models/qwen25-3b-ft-q8_0.gguf",
              "models/qwen25-3b-ft-q4_k_m.gguf"],
             "convert_hf_to_gguf + llama-quantize (见 README 阶段5)"),
            ("速度基准 llama-bench", ["experiment:bench"],
             "python pipeline/bench_quant.py --label <名字>"),
            ("PPL 评测", ["experiment:eval_qwen"],
             "python pipeline/eval.py --models models/qwen25-3b-ft-q4_k_m.gguf --label qwen-eval"),
            ("imatrix 方案对比", ["experiment:sweep_qwen"],
             "python pipeline/quantize_sweep.py --input models/qwen25-3b-ft-f16.gguf --label qwen-sweep"),
            ("投机解码 (draft=Qwen2.5-0.5B)", ["experiment:spec"],
             "python pipeline/spec_decode.py --draft models/qwen2.5-0.5b-instruct-q4_k_m.gguf "
             "--target models/qwen25-3b-ft-q4_k_m.gguf --ngl 99 --server <bin-cuda/llama-server>"),
        ],
    },
    {
        "line": "跨线实验",
        "cells": [
            ("投机解码 (draft+target)", ["models/tinyllm-target-q4_k_m.gguf"],
             "先 make train-target 训练 target 模型, 再 make spec"),
            ("跨实验对比 compare", ["experiment:__any__"],
             "python pipeline/compare.py --baseline <label>"),
        ],
    },
]


def experiments_on_disk() -> dict[str, list[dict]]:
    """experiment 名 -> 该名下的结构化记录列表。"""
    found: dict[str, list[dict]] = {}
    if EXPERIMENTS_DIR.exists():
        for f in sorted(EXPERIMENTS_DIR.glob("*.json")):
            try:
                doc = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            found.setdefault(doc.get("experiment", "?"), []).append(
                {"label": doc.get("label"), "file": f.name})
    return found


def cell_status(needles: list[str], exps: dict) -> tuple[str, str]:
    """返回 (状态, 证据说明)。 needles 里任一条件满足即算完成。"""
    hits = []
    for n in needles:
        if n.startswith("experiment:"):
            spec = n.split(":", 1)[1]
            if spec == "__any__":
                if exps:
                    hits.append(f"{sum(len(v) for v in exps.values())} 条实验记录")
                continue
            base, _, tag = spec.partition("_")
            docs = list(exps.get(base, []))
            if tag:
                docs = [e for e in docs if tag in (e["label"] or "").lower()]
            if docs:
                labels = ", ".join(e["label"] for e in docs)
                hits.append(f"实验记录 {base} ({labels})")
        else:
            p = ROOT / n
            if p.exists():
                hits.append(n.rstrip("/"))
    return (DONE, "; ".join(hits)) if hits else (TODO, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", help="只打印不写报告")
    args = ap.parse_args()

    exps = experiments_on_disk()
    n_done = 0
    n_all = 0
    sections = []
    for group in MATRIX:
        rows = []
        for stage, needles, howto in group["cells"]:
            n_all += 1
            status, evidence = cell_status(needles, exps)
            n_done += status == DONE
            rows.append([stage, status, evidence or "—", howto])
        sections.append((group["line"], markdown_table(
            ["阶段", "状态", "证据 (产物/实验记录)", "怎么跑"], rows)))

    lines = ["",
             f"_矩阵由 pipeline/matrix.py 扫描磁盘产物与 out/experiments/ 自动生成, "
             f"不依赖手工登记。_",
             ""]
    for title, table in sections:
        lines += [f"## {title}", "", table, ""]
    lines += ["## 结构化实验记录", ""]
    if exps:
        rows = [[name, ", ".join(e["label"] for e in docs)]
                for name, docs in sorted(exps.items())]
        lines += [markdown_table(["实验类型", "label"], rows), ""]
    else:
        lines += ["还没有。跑任何基准时加 `--label <名字>` 即自动生成。", ""]
    lines += ["> 约定: 每跑一格就带上 `--label`, 新对比用 "
              "`python pipeline/compare.py --baseline <label>` 汇总。"]

    text = "\n".join(lines)
    if args.print:
        print(text)
        return 0

    from common import write_report
    out = write_report(ROOT / "out" / "reports" / "matrix.md",
                       f"实验矩阵 — 完成 {n_done}/{n_all} 格", text)
    _register_report("matrix", out, "实验矩阵", True)
    print(f"[matrix] 完成 {n_done}/{n_all} 格")
    return 0


if __name__ == "__main__":
    sys.exit(main())
