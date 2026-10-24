# 实验数据包（QFPSI v6）

由 `experiments/make_dataset.py` 自动生成。每个 JSON 一份 CSV（`results/csv/`），
`MANIFEST.json` 记录脚本、行数、大小和 sha256 前缀。

| 数据文件 | 行数 | 大小 | sha256 | 脚本 | 测什么 |
|---|---|---|---|---|---|
| `MANIFEST.json` | 1 | 3 KiB | `4fd7315ea151` | `(unrecorded)` |  |
| `ablations.json` | 18 | 5 KiB | `2425e37152e8` | `run_ablations.py` | Boundary expansion r, token count T, dimensionality D (calibration and test sets disjoint) |
| `adaptive_proto.json` | 3 | 0 KiB | `dc39b092db35` | `proto/adaptive_proto.py` | Executed Adaptive-B: coarse feedback, calibrated width policy, three fine indexes |
| `boundary.json` | 13 | 12 KiB | `a9add7bb1c42` | `run_boundary.py` | Separation of genuine (0.85) from near-miss (0.80/0.82) pairs; 13 configs |
| `oprf_bench.json` | 1 | 0 KiB | `bb3c6e5b327f` | `proto/bench_oprf.py` | Ristretto255 OPRF microbenchmark: scalar mult, blind, evaluate, unblind |
| `predicate_multiseed.json` | 1 | 16 KiB | `aa11a6131561` | `run_multiseed.py` | Token overlap vs cosine similarity; 12 (B,w) configs, 5 rotations |
| `proto.json` | 1 | 0 KiB | `2adc1a6ecfbb` | `proto/qfpsi_proto.py` | Fixed-B prototype, N=1000, loopback |
| `proto_10k.json` | 1 | 0 KiB | `a900dfbf186f` | `proto/qfpsi_proto.py` | Fixed-B prototype, N=10000, loopback |
| `proto_lan.json` | 1 | 0 KiB | `3314998684ff` | `proto/qfpsi_proto.py` | Fixed-B prototype under emulated fast LAN |
| `proto_wan.json` | 1 | 0 KiB | `240c268abcdd` | `proto/qfpsi_proto.py` | Fixed-B prototype under emulated slow WAN |
| `real_sift.json` | 3 | 1 KiB | `72dc5583e6bc` | `run_realdata.py` | Accuracy on SIFT1M with the official 100-NN ground truth; (B,w,T)=(2,2,128) |
| `real_sift_w4.json` | 3 | 1 KiB | `12d20aa64b6d` | `run_realdata.py` | SIFT1M accuracy at (2,4,128) |
| `real_sift_w8.json` | 3 | 1 KiB | `ee9536d88f0c` | `run_realdata.py` | SIFT1M accuracy at (2,8,512) |
| `selectivity.json` | 8 | 8 KiB | `39723e92904d` | `run_selectivity.py` | Accuracy vs selectivity trade-off (token space, bucket size, near-miss rate) |
| `transfer.json` | 1 | 1 KiB | `e5819e8d73ff` | `run_transfer.py` | Release probability vs cosine for thresholds calibrated at 99% recall |
| `twostage.json` | 16 | 3 KiB | `3aeb8e5a6181` | `run_twostage.py` | Count-only simulation of the two-stage protocol over t_c, i.i.d. and clustered databases |
| `twostage_proto.json` | 2 | 0 KiB | `428726557150` | `proto/twostage_proto.py` | Executed two-stage protocol: OPRF, packed indexes, Shamir 61-of-128, reconstruction |

## 论文表格 → 数据文件

| 论文位置 | 数据文件 | CSV |
|---|---|---|
| 表 2 令牌重叠与工作点 | `predicate_multiseed.json` | `csv/predicate_multiseed.csv` |
| 表 3 消融（r / T / D） | `ablations.json` | `csv/ablations.csv` |
| 表 4 近失配分辨率 | `boundary.json` | `csv/boundary.csv` |
| 图 6 转移曲线 | `transfer.json` | `csv/transfer.csv` |
| 表 5 两阶段执行 | `twostage_proto.json` | `csv/twostage_proto.csv` |
| §6.3 正文（精度-选择度网格，未列表） | `selectivity.json` | `csv/selectivity.csv` |
| 表 7 原型测量 | `oprf_bench.json`, `proto_10k.json`, `proto_lan.json`, `proto_wan.json` | 同名 CSV |
| 表 8 FLPSI 对照（引用值） | 无本地数据，转录自 Uzun et al. Fig. 8 | — |

## 复现

```bash
cd ~/Desktop/论文-修改
.venv-exp/bin/python experiments/sim/run_multiseed.py
.venv-exp/bin/python experiments/sim/run_ablations.py
.venv-exp/bin/python experiments/sim/run_boundary.py
.venv-exp/bin/python experiments/sim/run_transfer.py
.venv-exp/bin/python experiments/sim/run_selectivity.py
.venv-exp/bin/python experiments/sim/run_twostage.py
brew install libsodium     # 原型依赖
.venv-exp/bin/python experiments/proto/bench_oprf.py
.venv-exp/bin/python experiments/proto/qfpsi_proto.py --n 10000 --queries 100
.venv-exp/bin/python experiments/proto/qfpsi_proto.py --n 10000 --shaping fast
.venv-exp/bin/python experiments/proto/qfpsi_proto.py --n 10000 --shaping slow
.venv-exp/bin/python experiments/proto/twostage_proto.py --n 10000 --q 40
.venv-exp/bin/python experiments/make_dataset.py   # 重新生成 CSV 与本文本
```
