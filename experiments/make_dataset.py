"""
Build the experiment data package: CSV exports + a manifest + SUMMARY.md.

Every JSON in experiments/results/ is flattened to a CSV (one row per record),
the manifest records which script produced it and what it measures, and
SUMMARY.md is regenerated so the paper's tables can be traced back to files.

Run:  python experiments/make_dataset.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
CSV = RES / "csv"

PROVENANCE = {
    "predicate_multiseed.json": (
        "run_multiseed.py",
        "Token overlap vs cosine similarity; 12 (B,w) configs, 5 rotations"),
    "ablations.json": (
        "run_ablations.py",
        "Boundary expansion r, token count T, dimensionality D (calibration and test sets disjoint)"),
    "boundary.json": (
        "run_boundary.py",
        "Separation of genuine (0.85) from near-miss (0.80/0.82) pairs; 13 configs"),
    "transfer.json": (
        "run_transfer.py",
        "Release probability vs cosine for thresholds calibrated at 99% recall"),
    "twostage.json": (
        "run_twostage.py",
        "Count-only simulation of the two-stage protocol over t_c, i.i.d. and clustered databases"),
    "real_sift.json": (
        "run_realdata.py",
        "Accuracy on SIFT1M with the official 100-NN ground truth; (B,w,T)=(2,2,128)"),
    "real_sift_w4.json": (
        "run_realdata.py", "SIFT1M accuracy at (2,4,128)"),
    "real_sift_w8.json": (
        "run_realdata.py", "SIFT1M accuracy at (2,8,512)"),
    "adaptive_proto.json": (
        "proto/adaptive_proto.py",
        "Executed Adaptive-B: coarse feedback, calibrated width policy, three fine indexes"),
    "twostage_proto.json": (
        "proto/twostage_proto.py",
        "Executed two-stage protocol: OPRF, packed indexes, Shamir 61-of-128, reconstruction"),
    "selectivity.json": (
        "run_selectivity.py",
        "Accuracy vs selectivity trade-off (token space, bucket size, near-miss rate)"),
    "oprf_bench.json": (
        "proto/bench_oprf.py",
        "Ristretto255 OPRF microbenchmark: scalar mult, blind, evaluate, unblind"),
    "proto.json": ("proto/qfpsi_proto.py", "Fixed-B prototype, N=1000, loopback"),
    "proto_10k.json": ("proto/qfpsi_proto.py", "Fixed-B prototype, N=10000, loopback"),
    "proto_lan.json": ("proto/qfpsi_proto.py", "Fixed-B prototype under emulated fast LAN"),
    "proto_wan.json": ("proto/qfpsi_proto.py", "Fixed-B prototype under emulated slow WAN"),
}


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}{i}."))
    elif isinstance(obj, list):
        out[prefix.rstrip(".")] = json.dumps(obj)
    else:
        out[prefix.rstrip(".")] = obj
    return out


def main():
    CSV.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(RES.glob("*.json")):
        data = json.loads(f.read_text())
        script, desc = PROVENANCE.get(f.name, ("(unrecorded)", ""))
        # one CSV row per top-level record, wide format
        if isinstance(data, dict) and ("rows" in data or "results" in data):
            items = data.get("rows") or data.get("results")
            if isinstance(items, dict):
                items = list(items.values())
            table = [flatten(it) for it in items]
            keys = sorted({k for r in table for k in r})
            out = CSV / (f.stem + ".csv")
            with out.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                for r in table:
                    w.writerow(r)
            nrows = len(table)
        else:
            out = CSV / (f.stem + ".csv")
            with out.open("w", newline="") as fh:
                w = csv.writer(fh)
                for k, v in flatten(data).items():
                    w.writerow([k, v])
            nrows = 1
        rows.append({
            "file": f.name, "csv": out.name, "rows": nrows,
            "bytes": f.stat().st_size,
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest()[:12],
            "script": script, "measures": desc,
        })

    manifest = RES / "MANIFEST.json"
    manifest.write_text(json.dumps(rows, indent=1))

    lines = [
        "# 实验数据包（QFPSI v6）",
        "",
        "由 `experiments/make_dataset.py` 自动生成。每个 JSON 一份 CSV（`results/csv/`），",
        "`MANIFEST.json` 记录脚本、行数、大小和 sha256 前缀。",
        "",
        "| 数据文件 | 行数 | 大小 | sha256 | 脚本 | 测什么 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| `{r['file']}` | {r['rows']} | {r['bytes']//1024} KiB | `{r['sha256']}` "
                     f"| `{r['script']}` | {r['measures']} |")

    lines += ["", "## 论文表格 → 数据文件", "",
              "| 论文位置 | 数据文件 | CSV |",
              "|---|---|---|",
              "| 表 2 令牌重叠与工作点 | `predicate_multiseed.json` | `csv/predicate_multiseed.csv` |",
              "| 表 3 消融（r / T / D） | `ablations.json` | `csv/ablations.csv` |",
              "| 表 4 近失配分辨率 | `boundary.json` | `csv/boundary.csv` |",
              "| 图 6 转移曲线 | `transfer.json` | `csv/transfer.csv` |",
              "| 表 5 两阶段执行 | `twostage_proto.json` | `csv/twostage_proto.csv` |",
              "| §6.3 正文（精度-选择度网格，未列表） | `selectivity.json` | `csv/selectivity.csv` |",
              "| 表 7 原型测量 | `oprf_bench.json`, `proto_10k.json`, `proto_lan.json`, `proto_wan.json` | 同名 CSV |",
              "| 表 8 FLPSI 对照（引用值） | 无本地数据，转录自 Uzun et al. Fig. 8 | — |",
              "", "## 复现", "", "```bash",
              "cd ~/Desktop/论文-修改",
              ".venv-exp/bin/python experiments/sim/run_multiseed.py",
              ".venv-exp/bin/python experiments/sim/run_ablations.py",
              ".venv-exp/bin/python experiments/sim/run_boundary.py",
              ".venv-exp/bin/python experiments/sim/run_transfer.py",
              ".venv-exp/bin/python experiments/sim/run_selectivity.py",
              ".venv-exp/bin/python experiments/sim/run_twostage.py",
              "brew install libsodium     # 原型依赖",
              ".venv-exp/bin/python experiments/proto/bench_oprf.py",
              ".venv-exp/bin/python experiments/proto/qfpsi_proto.py --n 10000 --queries 100",
              ".venv-exp/bin/python experiments/proto/qfpsi_proto.py --n 10000 --shaping fast",
              ".venv-exp/bin/python experiments/proto/qfpsi_proto.py --n 10000 --shaping slow",
              ".venv-exp/bin/python experiments/proto/twostage_proto.py --n 10000 --q 40",
              ".venv-exp/bin/python experiments/make_dataset.py   # 重新生成 CSV 与本文本",
              "```", ""]
    (RES / "SUMMARY.md").write_text("\n".join(lines))
    print(f"wrote {len(rows)} CSV files, MANIFEST.json and SUMMARY.md")


if __name__ == "__main__":
    main()
