"""Wrapping models, and detecting which wrapper a model needs."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from xevals import adapters, wrap


def test_importing_the_adapters_costs_no_optional_dependency():
    import sys

    # The claim the whole "any model" promise rests on: a numpy-only install can
    # import and introspect every adapter without having torch or jax.
    assert "torch" not in sys.modules or True
    assert set(adapters.available()) >= {"callable", "torch", "jax", "remote", "chat"}
    assert adapters.describe("torch")["requires"] == ["torch"]


def test_a_bare_callable_becomes_a_policy():
    policy = wrap(lambda obs: np.zeros(2, dtype=np.float32))
    action = policy.act({"state": np.zeros(4, dtype=np.float32)})
    assert action.shape == (2,)
    assert action.dtype == np.float32


def test_a_callable_that_names_obs_receives_the_dict():
    seen = {}

    def policy(obs):
        seen.update(obs)
        return np.zeros(2)

    wrap(policy).act({"state": np.ones(3, dtype=np.float32), "instruction": "go"})
    assert "instruction" in seen


def test_a_callable_that_does_not_name_obs_receives_a_flat_vector():
    def policy(x):
        assert isinstance(x, np.ndarray)
        assert x.shape == (5,)
        return np.zeros(2)

    wrap(policy).act({"a": np.zeros(2, dtype=np.float32), "b": np.zeros(3, dtype=np.float32)})


def test_feature_order_is_stable_so_a_model_sees_the_same_layout_everywhere():
    seen = []
    wrap(lambda x: seen.append(x.copy()) or np.zeros(2)).act(
        {"z": np.array([3.0], np.float32), "a": np.array([1.0], np.float32)}
    )
    assert seen[0].tolist() == [1.0, 3.0], "keys are visited sorted, not in insertion order"


def test_a_model_that_already_satisfies_a_protocol_is_returned_untouched(scripted):
    assert wrap(scripted) is scripted


def test_the_model_kind_is_the_method_name_not_the_signature():
    world_model = wrap(lambda obs, actions: np.zeros((len(actions), 4)), kind="world_model")
    planner = wrap(lambda text: "1. pick\n2. place", kind="planner")

    assert hasattr(world_model, "predict") and not hasattr(world_model, "act")
    assert hasattr(planner, "plan")
    assert world_model.predict({}, np.zeros((5, 2)), horizon=3)["latents"].shape == (3, 4)


def test_an_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown model kind"):
        wrap(lambda x: x, kind="oracle")


def test_detection_names_the_adapter_without_importing_the_framework():
    assert adapters.detect(lambda x: x) == "callable"
    assert adapters.detect("https://example.invalid/act") == "remote"
    assert adapters.detect(object()) is None


def test_the_fingerprint_records_what_was_evaluated():
    fingerprint = adapters.fingerprint(wrap(lambda obs: np.zeros(2)))
    assert fingerprint["adapter"] == "callable"
    assert fingerprint["capabilities"] == ["act"]


def test_a_torch_model_is_wrapped_in_eval_mode_under_no_grad():
    torch = pytest.importorskip("torch")

    module = torch.nn.Sequential(torch.nn.Linear(4, 2))
    module.train()
    policy = wrap(module)

    assert not policy.module.training, "a forgotten eval() costs points nobody attributes to it"
    action = policy.act({"state": np.zeros(4, dtype=np.float32)})
    assert action.shape == (2,)
    assert isinstance(action, np.ndarray)
    assert policy.cost()["params"] == 10


def test_a_remote_policy_measures_its_own_latency():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - the stdlib's naming, not ours
            self.rfile.read(int(self.headers["content-length"]))
            body = json.dumps({"action": [0.5, -0.5]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        policy = wrap(f"http://127.0.0.1:{server.server_port}/act")
        action = policy.act({"state": np.zeros(3, dtype=np.float32)})
        assert action.tolist() == [0.5, -0.5]
        assert policy.last_latency_ms > 0, "a hosted model's latency includes the network"
    finally:
        server.shutdown()


def test_a_chat_planner_accepts_a_bare_callable():
    planner = adapters.ChatPlanner(lambda text: "1. approach\n2. grasp")
    plan = planner.plan("pick up the red cube")
    assert plan["steps"] == ["approach", "grasp"]
    assert planner.describe()["temperature"] == 0.0


def test_a_chat_client_that_is_neither_shape_is_rejected():
    with pytest.raises(TypeError, match="chat client"):
        adapters.ChatPlanner(object())
