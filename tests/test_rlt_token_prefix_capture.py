from types import SimpleNamespace

import pytest
import torch

from rlt_so101_dual.adapters.lerobot.policies.modeling_rlt_token import RLTokenPolicy


class _FakePaligemmaWithExpert:
    def __init__(self, prefix: torch.Tensor | None) -> None:
        self.prefix = prefix
        self.forward_calls = 0

    def forward(self, *args, **kwargs):
        self.forward_calls += 1
        return [self.prefix, None], None


class _FakePI05:
    def __init__(self, prefix: torch.Tensor | None) -> None:
        self.model = SimpleNamespace(paligemma_with_expert=_FakePaligemmaWithExpert(prefix))

    def forward(self, batch, reduction: str = "mean"):
        self.model.paligemma_with_expert.forward(inputs_embeds=[torch.empty(1, 1, 1), None])
        return torch.tensor(1.0), {"loss": 1.0, "reduction": reduction}


def _policy_with_fake_pi05(prefix: torch.Tensor | None) -> RLTokenPolicy:
    policy = RLTokenPolicy.__new__(RLTokenPolicy)
    object.__setattr__(policy, "_pi05", _FakePI05(prefix))
    return policy


def test_forward_pi05_with_prefix_captures_hidden_states_and_restores_forward():
    expected = torch.randn(2, 4, 8)
    policy = _policy_with_fake_pi05(expected)
    target = policy._pi05.model.paligemma_with_expert
    original_forward = target.forward

    loss, info, prefix = policy._forward_pi05_with_prefix({}, reduction="none")

    assert loss.item() == pytest.approx(1.0)
    assert info["reduction"] == "none"
    assert prefix is expected
    assert target.forward.__self__ is original_forward.__self__
    assert target.forward.__func__ is original_forward.__func__


def test_forward_pi05_with_prefix_requires_prefix_output():
    policy = _policy_with_fake_pi05(None)

    with pytest.raises(RuntimeError, match="prefix hidden states"):
        policy._forward_pi05_with_prefix({}, reduction="mean")
