# rlt_so101_dual

**English** | [中文](README_zh.md)

**RL Token (RLT)** on **SO101 dual-arm**, using **Hugging Face LeRobot + π0.5**.

SO101 dual-arm scenario / demo adaptation of the RL Token method ([Physical Intelligence](https://www.pi.website/research/rlt), arXiv [2604.23073](https://arxiv.org/abs/2604.23073)). Not an official PI release.

License: Apache-2.0 · [`LICENSE`](LICENSE) / [`NOTICE`](NOTICE)

---

## Demo videos

### Successful case

https://github.com/user-attachments/assets/4d54dbf1-9824-471b-91f6-3ea72269602a

### Failed case

https://github.com/user-attachments/assets/4be130b2-9381-404e-81cc-ce5aa3d7d9ee

---

## What you get


| Stage                  | CLI                             |
| ---------------------- | ------------------------------- |
| Hardware check         | `rlt-so101-dual-preflight`      |
| Demo / VLA eval record | `rlt-so101-dual-record`         |
| RL Token (Stage B)     | `rlt-so101-dual-train-rl-token` |
| Real-robot online RL   | `rlt-so101-dual-online-train`   |

```text
preflight → record(teleop) → π0.5 SFT → train-rl-token → online-train
```

```text
obs → π0.5 → tokens + ref_chunk → RLToken → z_rl
    → state=[z_rl‖proprio] → ChunkActor → exec chunk (critical phase)
```

---

## Requirements

- Linux, Python ≥ 3.12
- LeRobot **0.5.1** with π0.5 extras + Feetech SDK
- SO101 **bimanual** (2 followers + 2 leaders) + 3 cameras (`left_wrist`, `right_wrist`, `right_front`)
- GPU: capture PC can run online inference; full π0.5 SFT wants a large-GPU training machine

---

## Quick start

```bash
git clone <this-repo> rlt_so101_dual
cd rlt_so101_dual

# Create/activate a conda (or venv) with LeRobot 0.5.1 + feetech, then:
pip install -e ".[dev]"

# Hardware: copy template and fill serial by-id + camera indices
cp configs/hardware/so101_dual_manifest.example.json \
   configs/hardware/so101_dual_manifest.json
# Edit ports with: ls /dev/serial/by-id/

rlt-so101-dual-preflight --setup-json configs/hardware/so101_dual_manifest.json
```

Calibrate / teleop with upstream LeRobot (`lerobot-calibrate`, `lerobot-teleoperate`) — see [`configs/hardware/README.md`](configs/hardware/README.md).

### Record demos (writes RLT `complementary_info`)

```bash
rlt-so101-dual-record full \
  --initial-source teleop \
  --setup-json configs/hardware/so101_dual_manifest.json \
  --dataset-tag so101_dual_demo \
  --num-episodes 30 --fps 30 \
  --discard-unlabeled-episodes \
  --task "YOUR TASK INSTRUCTION (keep identical through SFT / token / online)"
```

### Online RL (after SFT + RL Token)

```bash
rlt-so101-dual-online-train --help
```

**Do not** pass `--config` to `rlt-so101-dual-online-train`. Always pass `--gamma 0.999` at 30 fps. Defaults match the RLT paper recipe (absolute actor, `fixed_std=0.05`, sparse reward, RTC off).

---

## Data contract

Authoritative: [`src/rlt_so101_dual/core/shape_contract.py`](src/rlt_so101_dual/core/shape_contract.py). Do not change after you start collecting.

```text
bi_so_follower · 30 fps · action/state [12] (left_* + right_*)
images: left_wrist / right_wrist / right_front  →  π0.5 slots
RL chunk C=10 · VLA H=50 · RL token dim=2048
```


| Config     | Path                                                               |
| ---------- | ------------------------------------------------------------------ |
| Hardware   | `configs/hardware/so101_dual_manifest.json` (from `.example.json`) |
| Rename map | `configs/rename_maps/so101_dual.json`                              |
| RLT yaml   | `configs/rlt/so101_dual_rlt.yaml`                                  |

---

## Safety

This software is **not** a hardware e-stop. Keep a hand on the leaders / power when running policy or online RL on a real robot.

---

## Naming


| Kind                 | Name                   |
| -------------------- | ---------------------- |
| Package              | `rlt_so101_dual`       |
| CLI                  | `rlt-so101-dual-*`     |
| LeRobot`policy.type` | `rlt_token` / `rlt_ac` |

---

## Acknowledgments

This repository is based on:

- [Evo-RLT](https://github.com/MINT-SJTU/Evo-RLT) by [SJTU-MINT](https://github.com/MINT-SJTU)
- [RL Token](https://www.pi.website/research/rlt) by [Physical Intelligence](https://www.physicalintelligence.company/)
- [LeRobot](https://github.com/huggingface/lerobot) by [Hugging Face](https://github.com/huggingface)

Fill `configs/hardware/so101_dual_manifest.json` from the `.example.json` before real-robot use.
