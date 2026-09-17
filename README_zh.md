# rlt_so101_dual

[English](README.md) | **中文**

**RL Token (RLT)** on **SO101 dual-arm**，基于 **Hugging Face LeRobot + π0.5**。

RL Token 方法（[Physical Intelligence](https://www.pi.website/research/rlt)，arXiv [2604.23073](https://arxiv.org/abs/2604.23073)）在 SO101 双臂场景上的适配 / demo。**非** PI 官方发布。

配置：[`configs/README.md`](configs/README.md) · 数据约定：[`src/rlt_so101_dual/core/shape_contract.py`](src/rlt_so101_dual/core/shape_contract.py)  
许可证：Apache-2.0 · [`LICENSE`](LICENSE) / [`NOTICE`](NOTICE)

---

## 能做什么

| 阶段 | CLI |
| --- | --- |
| 硬件检查 | `rlt-so101-dual-preflight` |
| Demo / VLA 评测录制 | `rlt-so101-dual-record` |
| RL Token（Stage B） | `rlt-so101-dual-train-rl-token` |
| 真机 online RL | `rlt-so101-dual-online-train` |

```text
preflight → record(teleop) → π0.5 SFT → train-rl-token → online-train
```

```text
obs → π0.5 → tokens + ref_chunk → RLToken → z_rl
    → state=[z_rl‖proprio] → ChunkActor → exec chunk (critical phase)
```

---

## 环境要求

- Linux，Python ≥ 3.12  
- LeRobot **0.5.1**（含 π0.5 extras）+ Feetech SDK  
- SO101 **双臂**（2 从臂 + 2 主臂）+ 3 路相机（`left_wrist` / `right_wrist` / `right_front`）  
- GPU：采集机可做 online 推理；完整 π0.5 SFT 需要大显存训练机  

---

## 快速开始

```bash
git clone <this-repo> rlt_so101_dual
cd rlt_so101_dual

# 先准备好含 LeRobot 0.5.1 + feetech 的 conda/venv，然后：
pip install -e ".[dev]"

# 硬件：复制模板，填写串口 by-id 与相机编号
cp configs/hardware/so101_dual_manifest.example.json \
   configs/hardware/so101_dual_manifest.json
# 用 ls /dev/serial/by-id/ 编辑端口

rlt-so101-dual-preflight --setup-json configs/hardware/so101_dual_manifest.json
```

标定 / 遥操使用上游 LeRobot（`lerobot-calibrate`、`lerobot-teleoperate`）— 见 [`configs/hardware/README.md`](configs/hardware/README.md)。

### 录制 demo（写入 RLT `complementary_info`）

```bash
rlt-so101-dual-record full \
  --initial-source teleop \
  --setup-json configs/hardware/so101_dual_manifest.json \
  --dataset-tag so101_dual_demo \
  --num-episodes 30 --fps 30 \
  --discard-unlabeled-episodes \
  --task "YOUR TASK INSTRUCTION (keep identical through SFT / token / online)"
```

### Online RL（完成 SFT + RL Token 之后）

```bash
rlt-so101-dual-online-train --help
```

**不要**给 `rlt-so101-dual-online-train` 传 `--config`。30 fps 下请始终传 `--gamma 0.999`。默认对齐 RLT 论文配方（绝对 Actor、`fixed_std=0.05`、稀疏奖励、关 RTC）。

---

## 数据约定

权威定义：[`src/rlt_so101_dual/core/shape_contract.py`](src/rlt_so101_dual/core/shape_contract.py)。开始采集后不要再改。

```text
bi_so_follower · 30 fps · action/state [12] (left_* + right_*)
images: left_wrist / right_wrist / right_front  →  π0.5 slots
RL chunk C=10 · VLA H=50 · RL token dim=2048
```

| 配置 | 路径 |
| --- | --- |
| 硬件 | `configs/hardware/so101_dual_manifest.json`（由 `.example.json` 复制） |
| Rename map | `configs/rename_maps/so101_dual.json` |
| RLT yaml | `configs/rlt/so101_dual_rlt.yaml` |

---

## 安全

本软件**不是**硬件急停。真机跑策略或 online RL 时，手要能碰到主臂 / 电源。

---

## 命名

| 类型 | 名称 |
| --- | --- |
| 包名 | `rlt_so101_dual` |
| CLI | `rlt-so101-dual-*` |
| LeRobot `policy.type` | `rlt_token` / `rlt_ac` |

---

## Acknowledgments

本仓库基于：

- [Evo-RLT](https://github.com/MINT-SJTU/Evo-RLT) by [SJTU-MINT](https://github.com/MINT-SJTU)
- [RL Token](https://www.pi.website/research/rlt) by [Physical Intelligence](https://www.physicalintelligence.company/)
- [LeRobot](https://github.com/huggingface/lerobot) by [Hugging Face](https://github.com/huggingface)

真机使用前，请从 `.example.json` 填写 `configs/hardware/so101_dual_manifest.json`。
