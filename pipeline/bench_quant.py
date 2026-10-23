#!/usr/bin/env python
"""
量化加速基准: 用 llama-bench 测同一模型不同量化档位的 prompt / decode 速度

用法:
    python pipeline/bench_quant.py --llama-cpp D:/tools/llama.cpp
    python pipeline/bench_quant.py --gguf models/qwen25-3b-f16.gguf models/qwen25-3b-q4_k_m.gguf

产物:
    bench/bench.json        原始结果
    bench/bench.md          对比表
"""
import argparse
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import write_experiment  # noqa: E402


def find_binary(llama_cpp: Path, name: str) -> Path:
    candidates = [
        llama_cpp / "bin-cuda" / name,
        llama_cpp / "bin" / name,
        llama_cpp / name,
    ]
    for c in candidates:
        if c.exists():
            return c
    for p in llama_cpp.rglob(name):
        return p
    raise SystemExit(f"[bench] 找不到 {name}, 请检查 --llama-cpp 目录")


def run_bench(bench_exe: Path, models: list[Path], ngl: int, pp: int, tg: int, reps: int):
    cmd = [str(bench_exe), "-p", str(pp), "-n", str(tg), "-r", str(reps), "-ngl", str(ngl), "-o", "csv"]
    for m in models:
        cmd += ["-m", str(m)]
    print("[bench] " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        raise SystemExit(f"[bench] llama-bench 失败, exit={proc.returncode}")
    return proc.stdout


def parse_csv(text: str):
    rows = []
    reader = csv.DictReader(io.StringIO(text))
    for r in reader:
        rows.append(r)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llama-cpp", type=Path, default=Path("D:/tools/llama.cpp"))
    p.add_argument("--gguf", nargs="*", type=Path, help="不指定则自动找 models/*.gguf")
    p.add_argument("--ngl", type=int, default=99, help="offload 到 GPU 的层数, 99=全部")
    p.add_argument("--pp", type=int, default=512, help="prompt 处理的 token 数")
    p.add_argument("--tg", type=int, default=128, help="decode 生成的 token 数")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--out", type=Path, default=ROOT / "bench")
    p.add_argument("--label", default=None, help="实验标签: 写入 out/experiments/bench__<label>.json, 供 compare.py 对比")
    args = p.parse_args()

    if args.gguf:
        models = [m.resolve() for m in args.gguf]
    else:
        models = sorted(ROOT.glob("models/*.gguf"))
    models = [m for m in models if m.exists()]
    if not models:
        raise SystemExit("[bench] 没有找到 .gguf 文件")

    bench_exe = find_binary(args.llama_cpp, "llama-bench.exe")
    print(f"[bench] llama-bench: {bench_exe}")
    text = run_bench(bench_exe, models, args.ngl, args.pp, args.tg, args.reps)
    rows = parse_csv(text)

    # 整理: 每个模型一行, prompt / decode 各取 t/s
    # CSV 用 n_prompt / n_gen 区分测试类型 (pp / tg)
    by_model = {}
    for r in rows:
        name = Path(r.get("model_filename") or r.get("model") or "").name
        if not name:
            continue
        n_prompt = int(float(r.get("n_prompt") or 0))
        n_gen = int(float(r.get("n_gen") or 0))
        try:
            ts_f = float(r.get("avg_ts") or 0)
        except ValueError:
            ts_f = 0.0
        entry = by_model.setdefault(name, {
            "size": int(float(r.get("model_size") or 0)),
            "type": r.get("model_type", ""),
            "params": int(float(r.get("model_n_params") or 0)),
        })
        if n_gen > 0:
            entry["decode_ts"] = ts_f
        elif n_prompt > 0:
            entry["prompt_ts"] = ts_f

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "bench.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # 以 F16 为基准算加速比
    base = None
    for name in by_model:
        if "f16" in name.lower() or "fp16" in name.lower():
            base = by_model[name]
            break
    base_decode = base.get("decode_ts") if base else None

    lines = [
        f"# 量化加速基准 (ngl={args.ngl}, pp={args.pp}, tg={args.tg}, reps={args.reps})",
        "",
        "| 模型 | 类型 | 体积 | prompt t/s | decode t/s | decode 加速比 |",
        "|------|------|------|-----------|-----------|--------------|",
    ]
    for name, e in sorted(by_model.items(), key=lambda kv: -kv[1].get("decode_ts", 0)):
        size_gb = e["size"] / 1e9 if e["size"] else 0
        dec = e.get("decode_ts", 0)
        speedup = f"{dec / base_decode:.2f}x" if base_decode else "-"
        lines.append(
            f"| {name} | {e.get('type', '')} | {size_gb:.2f} GB | "
            f"{e.get('prompt_ts', 0):.1f} | {dec:.1f} | {speedup} |"
        )
    md = "\n".join(lines)
    (args.out / "bench.md").write_text(md + "\n", encoding="utf-8")
    print("\n" + md)

    exp_rows = []
    for name, e in sorted(by_model.items(), key=lambda kv: -kv[1].get("decode_ts", 0)):
        dec = e.get("decode_ts", 0)
        exp_rows.append({
            "model": name,
            "type": e.get("type", ""),
            "size_gb": round(e["size"] / 1e9, 3) if e["size"] else None,
            "prompt_ts": round(e["prompt_ts"], 1) if e.get("prompt_ts") else None,
            "decode_ts": round(dec, 1) if dec else None,
            "speedup_vs_f16": round(dec / base_decode, 3) if base_decode and dec else None,
        })
    write_experiment(
        "bench", label=args.label,
        config={"ngl": args.ngl, "pp": args.pp, "tg": args.tg, "reps": args.reps,
                "models": ",".join(m.name for m in models)},
        metrics={"n_models": len(exp_rows),
                 "best_decode_ts": max((r["decode_ts"] or 0) for r in exp_rows) if exp_rows else None},
        rows=exp_rows, key="model",
        artifacts=[args.out / "bench.json", args.out / "bench.md"],
    )


if __name__ == "__main__":
    sys.exit(main())
