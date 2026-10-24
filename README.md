# QFPSI 实验代码与数据（Experiments for the QFPSI paper）

[![quick-check](https://github.com/caopeng1124/qfpsi-experiments/actions/workflows/quickcheck.yml/badge.svg)](https://github.com/caopeng1124/qfpsi-experiments/actions/workflows/quickcheck.yml)
`quick-check` 每次推送都会在干净的 Ubuntu 机器上跑一遍全部实验并逐条核对论文数值。

本仓库包含论文 **QFPSI: Fuzzy Private Set Intersection for High-Dimensional
Vectors via Calibrated Quantized Token Matching** 的全部实验代码、测量脚本、
原始结果与数据包。所有数字都由这里的脚本产生，没有手工抄写。

仓库只包含实验部分；论文源码与导师批注不在其中。

---

## 0. 审稿人快速开始（Reviewer quick start）

```bash
git clone https://github.com/caopeng1124/qfpsi-experiments.git
cd qfpsi-experiments
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
brew install libsodium          # Debian/Ubuntu: sudo apt install libsodium23
./run_all.sh                    # ~5 分钟：跑全部 11 组实验并自检
```

`run_all.sh` 依次执行 E1–E11，把结果写到 `experiments/results_quick/`（**不会覆盖**
已发表的 `experiments/results/`），日志写到 `logs/`，图写到 `figures/`，最后用
`scripts/check_results.py` 把结果与论文引用的数值逐条比对。成功时输出：

```
   all 23 checks passed
   STATUS:   PASS
```

- 完整复现（5 个种子、原样本量，约 20 分钟）：`./run_all.sh --full`
- 真实数据实验（E7）：`./scripts/get_data.sh` 下载 SIFT1M（525 MB，支持断点续传）后再跑
- 每台机器的硬件/版本、每步耗时、预期输出、以及"哪个实验支撑论文哪句话"，见
  **[`docs/ARTIFACT.md`](docs/ARTIFACT.md)**
- 也可以在 GitHub Actions 的 `quick-check` 工作流里直接看最近一次的运行日志与产物

---

## 1. 快速开始

```bash
# Python 依赖
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 原型还需要系统库 libsodium（Ristretto255）
brew install libsodium        # macOS；Linux: apt install libsodium23

# 数据（真实数据实验用，约 525 MB）
mkdir -p data
curl -L -C - -o data/sift-128-euclidean.hdf5 \
     https://ann-benchmarks.com/sift-128-euclidean.hdf5
```

环境：macOS 27 / Apple M3（8 核、16 GB）、Python 3.14、numpy 2.5、
matplotlib 3.11、h5py 3.16、libsodium 1.0.22。

---

## 2. 目录

```
experiments/
├── pilot/                   # 最早的可行性验证（单种子，已被 sim/ 取代）
├── sim/                     # 精度层（纯 numpy，不需要密码学）
│   ├── qfpsi_sim.py         # 构造：归一化→随机旋转→B-bit 量化→窗口令牌→重叠
│   ├── run_multiseed.py     # E1 令牌重叠 vs 相似度（5 个旋转种子）
│   ├── run_ablations.py     # E2 边界扩展 r / 令牌数 T / 维度 D
│   ├── run_boundary.py      # E3 近失配（0.80/0.82）分辨率
│   ├── run_transfer.py      # E4 释放概率曲线（转移函数）
│   ├── run_selectivity.py   # E5 精度 vs 选择度（2^{Bw} 与桶大小）
│   ├── run_twostage.py      # E6 两阶段计数模拟（无密码学）
│   ├── run_realdata.py      # E7 真实 SIFT1M 精度
│   ├── make_figure.py       # 生成论文 Fig. 5
│   └── make_figure_transfer.py  # 生成论文 Fig. 6
├── proto/                   # 系统层（真实密码学 + 索引 + 网络）
│   ├── ristretto.py         # libsodium Ristretto255 的 ctypes 绑定
│   ├── bench_oprf.py        # E8 2HashDH OPRF 微基准 + 正确性
│   ├── qfpsi_proto.py       # E9 fixed-B 原型：建索引、TCP 往返、字节计数、LAN/WAN 整形
│   ├── twostage_proto.py    # E10 两阶段端到端执行（OPRF+索引+Shamir 重构）
│   ├── adaptive_proto.py    # E11 Adaptive-B 执行（校准策略）
│   └── netem_dnctl.sh       # 内核级整形脚本（需 root，可选）
├── results/                 # 所有原始结果
│   ├── *.json               # 每个实验一份
│   ├── csv/*.csv            # 同名扁平化导出
│   ├── MANIFEST.json        # 脚本、行数、大小、sha256 前缀
│   └── SUMMARY.md           # 数据索引：论文表 ↔ 数据文件 ↔ 复现命令
└── make_dataset.py          # 重新生成 CSV / MANIFEST / SUMMARY
```

---

## 3. 实验清单（问题 → 脚本 → 结果）

| ID | 问题 | 命令 | 耗时 | 结果文件 |
|---|---|---|---|---|
| E1 | 令牌重叠是否随相似度单调、可校准？ | `python experiments/sim/run_multiseed.py` | 5 s | `predicate_multiseed.json` |
| E2 | 边界扩展、令牌数、维度各影响多少？ | `python experiments/sim/run_ablations.py` | 30 s | `ablations.json` |
| E3 | 能否区分 0.85 与 0.80 的"近失配"？ | `python experiments/sim/run_boundary.py` | 3 min | `boundary.json` |
| E4 | 固定召回下，释放概率随相似度如何变化？ | `python experiments/sim/run_transfer.py` | 1 min | `transfer.json` |
| E5 | 选择度（桶大小）与精度的权衡 | `python experiments/sim/run_selectivity.py` | 4 min | `selectivity.json` |
| E6 | 两阶段候选集比例（计数模拟） | `python experiments/sim/run_twostage.py` | 2 min | `twostage.json` |
| E7 | 真实描述子上的精度 | `python experiments/sim/run_realdata.py --queries 800` | 3 min | `real_sift*.json` |
| E8 | OPRF 每次标量乘多快？ | `python experiments/proto/bench_oprf.py` | 20 s | `oprf_bench.json` |
| E9 | fixed-B 端到端延迟与字节 | `python experiments/proto/qfpsi_proto.py --n 10000 --queries 100` | 1 min | `proto_10k.json` |
| E9b | LAN/WAN 响应时间 | 加 `--shaping fast` / `--shaping slow` | 1 min | `proto_lan.json`, `proto_wan.json` |
| E10 | 两阶段全流程（含标签重构） | `python experiments/proto/twostage_proto.py --n 10000 --q 40` | 3 min | `twostage_proto.json` |
| E11 | Adaptive-B 是否有效？ | `python experiments/proto/adaptive_proto.py --n 5000 --q 32 --calib 20` | 3 min | `adaptive_proto.json` |

统一入口：`python experiments/make_dataset.py` 重新生成 CSV、MANIFEST 和
`results/SUMMARY.md`。

---

## 4. 主要结果

### 4.1 令牌重叠的解析预言（E1）

随机旋转后，坐标为联合高斯、相关系数 ρ；单比特编码的逐坐标一致概率为

  p_c(ρ) = 1/2 + (1/π)·arcsin ρ,

于是窗口令牌的期望重叠 ≈ T·p_c^w。在 ρ=0.85、B=1 时：

| w | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 预测 | 105.4 | 86.8 | 58.8 | 27.0 |
| 实测 | 105.5 | 86.8 | 58.9 | 26.9 |

### 4.2 边界分辨率是硬限制（E3/E4）

把阈值校准到"0.85 的对 99% 释放"后，0.80 的对仍有 **93–95%** 被释放，
d′ 在所有 13 个配置下都 **< 1**。这说明谓词是**软阈值**，过渡带宽约 0.15；
任何参数组合都救不回来（E5 再次确认）。

### 4.3 选择度与精度的矛盾（E5/E10）

索引按令牌建，令牌空间只有 2^{Bw}，因此每查询要扫 T·N/2^{Bw} 条索引：

| (B,w,T) | 桶大小@N=10⁶ | 扫描/查询@10⁶ | d′(远) | d′(近) | 0.80 释放率 |
|---|---|---|---|---|---|
| (2,2,128) | 62,500 | 8.0×10⁶ | 3.74 | 0.74 | 94.4% |
| (2,8,512) | 15 | 7.8×10³ | 3.25 | 0.93 | 92.8% |
| (2,12,512) | 0.1 | 31 | 2.26 | 0.84 | 94.2% |

### 4.4 真实数据（E7，SIFT1M）

| 配置 | 重叠 近/中/远 | FNMR@1%FMR | 近失配释放率 | d′(远) |
|---|---|---|---|---|
| (2,2,128) | 81.3 / 68.8 / 55.1 | 40% | 16% | 2.2 |
| (2,4,128) | 52 / 37 / 24 | 35% | 17% | 2.3 |
| (2,8,512) | 87 / 42 / 19 | 26% | 22% | 1.8 |

真实 SIFT 的"无关记录"重叠地板是 55/128（高斯模型预测 32/128），
因为描述子稀疏、大量坐标量化到同一码。**合成实验是上界，不是预测。**

### 4.5 系统原型（E8–E11，Apple M3 单核）

| 指标 | 数值 |
|---|---|
| Ristretto255 标量乘 | 32.5 μs（p95 36.7） |
| 单个在线令牌（3 次标量乘 + 2 次哈希） | 130 μs |
| 索引构建 N=10⁴（1.28M 条目，8 进程） | 12.6 s，74 MiB |
| 查询延迟（fixed-B, loopback） | 18.2 ms（p95 18.8） |
| 每查询字节（80 hits） | 13,986 B（上 8,202 / 下 5,784） |
| fast LAN（500 MB/s, 0.5 ms RTT） | 19.1 ms |
| slow WAN（40 MB/s, 34 ms RTT） | 100.3 ms |
| 两阶段执行（N=10⁴）：真匹配恢复 | 97%（i.i.d. 与近失配库） |
| Adaptive-B | 校准 θ=69，测试集退化为常数 → 相对固定 B=2 无收益 |

---

## 5. 复现注意事项

- **证据分级**：解析值（协议格式决定）、实测值（本仓库代码产生）、引用值
  （转录自他人论文）。`results/SUMMARY.md` 标明每个数字的来源。
- **合成 vs 真实**：E1–E6 用各向同性高斯对，用于机制与参数区间；只有 E7 用
  真实描述子。两者不可混用。
- **Proto 的局限**：Python 实现、单核、loopback 或用户态整形；内核级
  `dnctl`/`pfctl` 脚本在 `proto/netem_dnctl.sh`（需要 root）。
- **没有实现的**：恶意安全、PIR、GIST/Deep/face 数据集上的测量。

## 6. 数据来源

- SIFT1M：ANN-Benchmarks，`sift-128-euclidean.hdf5`（train 1M×128、test 10k×128、
  官方 100-NN ground truth）。下载地址见第 1 节。
- 其余实验不依赖外部数据（合成向量）。

## 7. 许可与引用

本仓库公开仅供复现与评审参考；如需在衍生工作中使用，请先联系作者。
论文引用信息将在正式发表后补上。
