#!/usr/bin/env python
"""
对比实验汇总: 把 out/experiments/ 下的结构化实验报告 (llm_go.experiment/1)
合并成一张横向对比表, 支持指定基线计算相对变化。

用法:
    python pipeline/compare.py                       # 汇总 out/experiments 下全部
    python pipeline/compare.py --files a.json b.json # 指定文件
    python pipeline/compare.py --baseline base-v1    # 相对该 label 算变化百分比
    python pipeline/compare.py --metrics decode_tok_s,ppl   # 只看指定指标

产物:
    out/comparison.md / .json          对比报告
    out/reports/comparison.md          同步一份进报告目录 (界面 📄 抽屉可见)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    EXPERIMENTS_DIR, REPORTS_DIR, _register_report, markdown_table,
    resolve_path, write_report,
)

SCHEMA = "llm_go.experiment/1"


def parse_args():
    p = argparse.ArgumentParser(description="结构化实验报告对比")
    p.add_argument("--files", nargs="*", default=None,
                   help="要对比的实验 JSON (默认扫 out/experiments/*.json)")
    p.add_argument("--metrics", default=None,
                   help="逗号分隔, 只看这些指标 (实验级与行级都生效)")
    p.add_argument("--baseline", default=None,
                   help="基线实验的 label, 行级数值列会附加相对它的变化百分比")
    p.add_argument("--out", default="out/comparison.md", help="对比报告输出路径")
    p.add_argument("--json-out", default="out/comparison.json", help="结构化对比输出路径")
    p.add_argument("--no-report-copy", action="store_true",
                   help="不同步到 out/reports/ (默认同步, 界面可见)")
    p.add_argument("--exp-dir", default=None, help="实验 JSON 目录 (默认 out/experiments)")
    return p.parse_args()


def load_docs(args) -> list[dict]:
    files = ([Path(f) for f in args.files] if args.files
             else sorted(Path(args.exp_dir).glob("*.json") if args.exp_dir
                         else EXPERIMENTS_DIR.glob("*.json")))
    docs = []
    for f in files:
        if not f.is_absolute():
            f = ROOT / f
        if not f.exists():
            print(f"[compare] 跳过不存在的文件: {f}")
            continue
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[compare] 跳过无法解析的 {f.name}: {e}")
            continue
        if doc.get("schema") != SCHEMA:
            print(f"[compare] 跳过不支持的 schema ({doc.get('schema') or '缺失'}): {f.name}")
            continue
        doc["_file"] = f.name
        docs.append(doc)
    if not docs:
        sys.exit("[compare] 没有可对比的实验报告。先跑带 --label 的基准脚本, "
                 "或用 --files 指定 JSON。")
    return docs


def fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def metric_filter(metrics: dict, allow: set[str] | None) -> dict:
    if not allow:
        return metrics
    return {k: v for k, v in metrics.items() if k in allow}


def headline_table(docs: list[dict], allow: set[str] | None) -> tuple[list[str], list[list]]:
    """实验级指标: 一行一个指标, 一列一个实验。"""
    keys: list[str] = []
    for d in docs:
        for k in metric_filter(d.get("metrics") or {}, allow):
            if k not in keys:
                keys.append(k)
    cols = [f"{d['label']}<br><sub>{d['experiment']} · {d.get('timestamp', '')}</sub>"
            for d in docs]
    rows = [[k] + [fmt((d.get("metrics") or {}).get(k)) for d in docs] for k in keys]
    return ["指标", *cols], rows


def row_metric_names(docs: list[dict], allow: set[str] | None) -> list[str]:
    names: list[str] = []
    for d in docs:
        for r in d.get("rows") or []:
            for k, v in r.items():
                if k == d.get("key"):
                    continue
                if allow and k not in allow:
                    continue
                if isinstance(v, (int, float, bool)) or v is None:
                    if k not in names:
                        names.append(k)
    return names


def row_tables(docs: list[dict], allow: set[str] | None,
               baseline: str | None) -> list[tuple[str, str, str]]:
    """按行主键 (key 列) 分组, 把多个实验的行合并成横向对比表。

    指定 --baseline 时, 数值单元格会附加相对基线同 key 同指标的百分比变化
    (基线列本身不显示变化; 值越大越好/越小越好不做区分, 只报正负)。
    返回 [(key, 标题, Markdown 表), ...]
    """
    base_doc = None
    if baseline:
        base_doc = next((d for d in docs if d.get("label") == baseline), None)
        if base_doc is None:
            print(f"[compare] 找不到 label={baseline} 的基线实验, 跳过相对变化")

    groups: dict[str, list[dict]] = {}
    for d in docs:
        key = d.get("key")
        if key and (d.get("rows") or []):
            groups.setdefault(key, []).append(d)

    out: list[tuple[str, str]] = []
    for key, group in groups.items():
        names = row_metric_names(group, allow)
        if not names:
            continue
        # 每个 (实验, key值) -> 行字典; 同 label 多条同 key 时后写覆盖前写
        cells: dict[tuple[str, str], dict] = {}
        for d in group:
            for r in d["rows"]:
                cells[(d["label"], str(r.get(key)))] = r
        key_vals = sorted({k for _, k in cells}, key=_natural)
        cols = [f"{d['label']}" for d in group]
        base_lookup = (({str(r.get(key)): r for r in base_doc.get("rows") or []}
                        if base_doc else {}))
        rows = []
        for kv in key_vals:
            row = [kv]
            for label in cols:
                r = cells.get((label, kv))
                if r is None:
                    row.append("—")
                    continue
                parts = []
                for m in names:
                    v = r.get(m)
                    txt = fmt(v)
                    if base_lookup and label != baseline:
                        bv = (base_lookup.get(kv) or {}).get(m)
                        if isinstance(v, (int, float)) and isinstance(bv, (int, float)) and bv:
                            txt += f" ({(v / bv - 1) * 100:+.1f}%)"
                    parts.append(f"{m}={txt}")
                row.append("<br>".join(parts))
            rows.append(row)
        title = f"行级对比 (主键: {key})"
        if baseline:
            title += f", 相对基线 {baseline}"
        out.append((key, title, markdown_table([key, *cols], rows)))
    return out


def _natural(s: str):
    return [int(t) if t.isdigit() else t for t in __import__("re").split(r"(\d+)", s)]


def main():
    args = parse_args()
    allow = ({m.strip() for m in args.metrics.split(",") if m.strip()}
             if args.metrics else None)
    docs = load_docs(args)
    print(f"[compare] 参与对比 {len(docs)} 份实验: "
          + ", ".join(f"{d['label']}({d['experiment']})" for d in docs))

    heads, head_rows = headline_table(docs, allow)
    tables = row_tables(docs, allow, args.baseline)

    body = "\n".join([
        f"对比了 **{len(docs)}** 份实验报告"
        + (f"，基线 `{args.baseline}`。" if args.baseline else "。"),
        "",
        "## 实验级指标",
        "",
        markdown_table(heads, head_rows),
        "",
        *[x for triple in tables for x in (f"## {triple[1]}", "", triple[2], "")],
        "",
        "> 结构化原始数据在 `out/experiments/`, 本文件的机器可读版在 "
        "`out/comparison.json`。",
    ])
    out_md = write_report(resolve_path(args.out), "实验对比报告", body)

    comparison = {
        "schema": "llm_go.comparison/1",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline": args.baseline,
        "experiments": [{"label": d["label"], "experiment": d["experiment"],
                         "timestamp": d.get("timestamp"), "file": d["_file"],
                         "config": d.get("config"), "metrics": d.get("metrics")}
                        for d in docs],
        "headline_rows": head_rows,
        "row_tables": [{"key": key, "title": title, "markdown": md}
                       for key, title, md in tables],
    }
    json_path = ROOT / args.json_out
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2),
                         encoding="utf-8", newline="\n")
    print(f"[compare] 结构化对比: {json_path.relative_to(ROOT).as_posix()}")

    if not args.no_report_copy:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        copy = REPORTS_DIR / Path(args.out).name if not Path(args.out).is_absolute() \
            else REPORTS_DIR / Path(args.out).name
        copy.write_text(out_md.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        _register_report("compare", copy, "实验对比报告", True)
        print(f"[compare] 已同步到报告目录: {copy.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
