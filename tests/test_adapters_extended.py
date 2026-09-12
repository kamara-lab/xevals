"""Regression cases for real adapter boundaries, without downloading checkpoints."""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from xevals import adapters, wrap


def test_callable_native_policy_keeps_its_interface():
    class Native:
        def act(self, obs, *, instruction=None):
            return np.zeros(2)

        def __call__(self, x):
            raise AssertionError("not the policy entrypoint")

    model = Native()
    assert wrap(model) is model
    assert adapters.detect(model) is None
    assert adapters.detect(wrap(lambda x: x)) is None


def test_specialised_detection_before_framework():
    module = type("Module", (), {"__module__": "torch.nn.modules.module"})
    lerobot = type("PreTrainedPolicy", (module,), {"__module__": "lerobot.policies.pretrained"})
    eqx = type("Module", (), {"__module__": "equinox"})
    xwm = type("Dynamics", (eqx,), {"__module__": "xwm.models"})
    assert adapters.detect(lerobot()) == "lerobot"
    assert adapters.detect(xwm()) == "xwm"


def test_unknown_and_incompatible_models_are_rejected():
    with pytest.raises(TypeError, match="cannot wrap"):
        wrap(object())
    torch = pytest.importorskip("torch")
    with pytest.raises(TypeError, match="model kind"):
        wrap(torch.nn.Linear(2, 2), kind="planner")


@pytest.mark.parametrize("value", [0.0, [], [[1, 2], [3, 4]], [[[1, 2]]]], ids=str)
def test_ambiguous_single_actions_rejected(value):
    with pytest.raises(ValueError, match="one action"):
        wrap(lambda x: value).act({"state": np.zeros(2)})


@pytest.mark.parametrize("value", [[np.nan, 0], [np.inf, 0], [0, 0, 0]])
def test_action_finiteness_and_declared_dimension(value):
    with pytest.raises(ValueError):
        wrap(lambda x: value, action_dim=2).act({"state": np.zeros(2)})


def test_singleton_batch_and_owned_action():
    values = np.ones((1, 2), dtype=np.float32)
    out = wrap(lambda x: values, action_dim=2).act({"state": np.zeros(2)})
    assert out.shape == (2,)
    values[:] = 9
    np.testing.assert_array_equal(out, [1, 1])


def test_explicit_feature_keys_cannot_disappear():
    with pytest.raises(ValueError, match="required numeric feature"):
        wrap(lambda x: [0, 0], feature_keys=("missing",)).act({"state": [1]})


@pytest.mark.parametrize("dtype", ["float64", "bfloat16"])
def test_torch_dtype_device_and_batched_parity(dtype):
    torch = pytest.importorskip("torch")
    module = torch.nn.Linear(3, 2, dtype=getattr(torch, dtype))
    policy = wrap(module, batch_mode="stateless", feature_keys=("state",))
    assert policy.device == "cpu"
    obs = [{"state": np.array([1, 2, 3], np.float32)}, {"state": np.zeros(3, np.float32)}]
    actual = policy.act_batch(obs)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, np.stack([policy.act(o) for o in obs]), atol=1e-6)
    assert "act_batch" in adapters.fingerprint(policy)["capabilities"]
    assert "act_batch" not in adapters.fingerprint(wrap(torch.nn.Linear(3, 2)))["capabilities"]


def test_torch_dict_keeps_integer_dtype_and_encode_payload():
    torch = pytest.importorskip("torch")

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("offset", torch.zeros(2, dtype=torch.float64))

        def forward(self, obs):
            assert obs["ids"].dtype == torch.int64
            assert obs["state"].dtype == torch.float64
            return obs["state"] + self.offset

        encode = forward

    p = wrap(Net())
    obs = {"state": np.ones(2, np.float32), "ids": np.array([1, 2], np.int64)}
    np.testing.assert_array_equal(p.act(obs), p.encode(obs))


def test_lerobot_dispatch_and_reset_use_select_action():
    torch = pytest.importorskip("torch")

    class FakeLeRobot(torch.nn.Module):
        def reset(self):
            self.calls = 0

        def select_action(self, batch):
            self.calls += 1
            assert batch["observation.image"].shape == (1, 3, 4, 4)
            assert batch["task"] == ["pick"]
            return torch.tensor([[self.calls, 0.0]])

        def forward(self, x):
            raise AssertionError("must call select_action")

    FakeLeRobot.__module__ = "lerobot.policies.fake"
    p = wrap(FakeLeRobot())
    assert isinstance(p, adapters.LeRobotPolicy)
    obs = {"image": np.zeros((4, 4, 3), np.uint8)}
    p.reset()
    assert p.act(obs, instruction="pick")[0] == 1
    assert p.act(obs, instruction="pick")[0] == 2
    p.reset()
    assert p.act(obs, instruction="pick")[0] == 1


def test_hf_preprocessing_eval_and_dtype(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("PIL")
    seen = {}

    class Model(torch.nn.Module):
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            seen.update(kwargs)
            return cls()

        def predict_action(self, **inputs):
            assert not self.training and not torch.is_grad_enabled()
            assert inputs["input_ids"].dtype == torch.int64
            assert inputs["pixel_values"].dtype == torch.bfloat16
            assert inputs["unnorm_key"] == "robot"
            return torch.ones(2, dtype=torch.bfloat16)

    class Processor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __call__(self, text, image, **kwargs):
            assert kwargs["return_tensors"] == "pt"
            assert "pick" in text
            return {"input_ids": torch.tensor([[1]]), "pixel_values": torch.zeros(1, 3, 4, 4)}

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoModelForVision2Seq=Model, AutoProcessor=Processor,
    ))
    p = adapters.HFVLAPolicy("fake", device="cpu", dtype="bfloat16", unnorm_key="robot")
    np.testing.assert_array_equal(p.act({"image": np.zeros((4, 4, 3), np.uint8)},
                                       instruction="pick"), [1, 1])
    assert seen["torch_dtype"] == torch.bfloat16


def test_torch_preserves_reset_for_serial_recurrent_modules():
    torch = pytest.importorskip("torch")

    class Recurrent(torch.nn.Module):
        def reset(self):
            self.count = 0

        def forward(self, x):
            self.count += 1
            return torch.full((1, 2), self.count)

    p = wrap(Recurrent())
    p.reset()
    assert p.act({"state": [0]})[0] == 1
    assert p.act({"state": [0]})[0] == 2
    p.reset()
    assert p.act({"state": [0]})[0] == 1


def test_explicit_adapter_overrides_native_detection():
    class Model:
        def act(self, obs, *, instruction=None):
            return [1, 1]

        def __call__(self, obs):
            return [2, 2]

    np.testing.assert_array_equal(wrap(Model(), adapter="callable").act({}), [2, 2])
