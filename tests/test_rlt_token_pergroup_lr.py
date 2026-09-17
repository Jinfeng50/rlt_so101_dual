from __future__ import annotations

from types import SimpleNamespace

import torch.nn as nn

from rlt_so101_dual.adapters.lerobot.policies.modeling_rlt_token import RLTokenPolicy


def _fake(vla_ft_weight, rl_lr, vla_lr, vla_trainable):
    rl = nn.Linear(4, 4)
    pi = nn.Linear(4, 4)
    for p in pi.parameters():
        p.requires_grad = vla_trainable
    cfg = SimpleNamespace(vla_ft_weight=vla_ft_weight, rl_token_lr=rl_lr, vla_lr=vla_lr)
    return SimpleNamespace(rl_token=rl, _pi05=pi, config=cfg)


def test_frozen_rlt_so101_dual_group():
    g = RLTokenPolicy.get_optim_params(_fake(0.0, 2e-4, 2e-5, False))
    assert len(g) == 1, g
    assert g[0]["lr"] == 2e-4
    print("ok frozen-vla: 1 group, rl_lr", g[0]["lr"])


def test_joint_two_groups_distinct_lr():
    fake = _fake(1.0, 2e-4, 2e-5, True)
    g = RLTokenPolicy.get_optim_params(fake)
    assert len(g) == 2, g
    assert g[0]["lr"] == 2e-4, g[0]["lr"]
    assert g[1]["lr"] == 2e-5, g[1]["lr"]
    rl_ids = {id(p) for p in fake.rl_token.parameters()}
    vla_ids = {id(p) for p in fake._pi05.parameters()}
    assert {id(p) for p in g[0]["params"]} == rl_ids
    assert {id(p) for p in g[1]["params"]} == vla_ids
    print("ok joint: rl_lr", g[0]["lr"], "vla_lr", g[1]["lr"], "groups distinct")


if __name__ == "__main__":
    test_frozen_rlt_so101_dual_group()
    test_joint_two_groups_distinct_lr()
    print("all per-group-lr tests passed")
