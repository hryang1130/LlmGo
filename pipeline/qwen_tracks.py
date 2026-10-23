#!/usr/bin/env python
"""
Qwen 双轨输出对比: 量化 (F16 vs Q4_K_M) × 微调 (base vs ft) 的 2x2 矩阵,
同一组 prompt 逐字对比输出 + 速度。

用法:
    python pipeline/qwen_tracks.py                       # 默认 4 模型 4 prompt
    python pipeline/qwen_tracks.py --max-tokens 192

产物:
    out/qwen_tracks.md            逐字输出对比报告 (同步 out/reports/)
    out/experiments/qwen_tracks__<label>.json   结构化结果
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    _register_report, http_json, markdown_table, write_experiment, write_report,
)
from eval import Server

MODELS = {
    "base-F16": "models/qwen25-3b-base-f16.gguf",
    "base-Q4": "models/qwen25-3b-base-q4_k_m.gguf",
    "ft-F16": "models/qwen25-3b-ft-f16.gguf",
    "ft-Q4": "models/qwen25-3b-ft-q4_k_m.gguf",
}

PROMPTS = [
    ("指令 · SFT 域", "什么是模型量化？请用两三句话解释。"),
    ("指令 · 常识", "地球绕太阳公转一圈大约需要多长时间？"),
    ("知识 · 解释", "为什么天空是蓝色的？"),
    ("创作 · 造句", "请用“秋天”写一句有意境的话。"),
]


def chat(srv_url: str, prompt: str, max_tokens: int) -> dict:
    data = http_json(f"{srv_url}/v1/chat/completions", {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0,
    }, timeout=900)
    text = data["choices"][0]["message"]["content"].strip()
    t = data.get("timings", {}) or data.get("usage", {})
    return {"text": text, "decode_ts": t.get("predicted_per_second"),
            "prompt_ts": t.get("prompt_per_second")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--label", default=None)
    ap.add_argument("--server", default="D:/tools/llama.cpp/bin-cuda/llama-server.exe")
    args = ap.parse_args()

    results: dict[str, list[dict]] = {}
    for name, rel in MODELS.items():
        model = ROOT / rel
        if not model.exists():
            print(f"[tracks] 跳过不存在的模型: {rel}")
            continue
        rows = []
        with Server(Path(args.server), model, args.ctx,
                    ["-ngl", str(args.ngl)], tag=f"tracks-{name}") as srv:
            for cap, prompt in PROMPTS:
                print(f"[tracks] {name} | {cap}", flush=True)
                r = chat(srv.url, prompt, args.max_tokens)
                rows.append({"prompt_tag": cap, "prompt": prompt, **r})
                print(f"    -> {r['decode_ts'] and round(r['decode_ts'], 1)} tok/s, "
                      f"{len(r['text'])} 字", flush=True)
        results[name] = rows

    # ---------- 报告 ----------
    lines = [
        f"测量条件: 同一组 {len(PROMPTS)} 个 prompt, 贪心解码 (temperature=0, "
        f"max_tokens={args.max_tokens}), -ngl {args.ngl}, 上下文 {args.ctx}; "
        "输出为逐字原文, 未做任何修饰。",
        "",
    ]
    for cap, prompt in PROMPTS:
        lines += [f"## Prompt: {cap}", "", f"> {prompt}", ""]
        for name in results:
            row = next((r for r in results[name] if r["prompt_tag"] == cap), None)
            if not row:
                continue
            speed = f"{row['decode_ts']:.1f}" if row.get("decode_ts") else "—"
            lines += [f"### {name} ({speed} tok/s)", "", "```text",
                      row["text"] or "(空输出)", "```", ""]
    # 汇总表
    lines += ["## 速度汇总 (decode tok/s)", "",
              markdown_table(["模型", *[c for c, _ in PROMPTS]],
                             [[name] + [f"{r['decode_ts']:.1f}" if r.get("decode_ts") else "—"
                                        for r in results[name]] for name in results]), ""]

    out = write_report(ROOT / "out" / "qwen_tracks.md",
                       "Qwen 双轨输出对比: 量化 × 微调", "\n".join(lines))
    copy = ROOT / "out" / "reports" / "qwen_tracks.md"
    copy.write_text(out.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    _register_report("qwen_tracks", copy, "Qwen 双轨输出对比", True)

    exp_rows = [{"model": name, "prompt_tag": r["prompt_tag"],
                 "decode_ts": r.get("decode_ts"), "chars": len(r["text"]),
                 "output": r["text"]} for name in results for r in results[name]]
    write_experiment("qwen_tracks", label=args.label,
                     config={"max_tokens": args.max_tokens, "ngl": args.ngl,
                             "ctx": args.ctx, "temperature": 0.0},
                     metrics={"n_models": len(results)},
                     rows=exp_rows, key="model",
                     artifacts=[out])
    print("[tracks] 完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
