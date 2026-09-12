"""Wrapping any model so it satisfies one of the protocols in :mod:`xevals.types`.

The load-bearing claim of this library is "it takes *any* model as input", and
this module is where that claim is either true or not. It is true by being
boring: :func:`wrap` looks at the object, picks an adapter, and hands back
something with an ``act``, ``predict`` or ``plan`` method. Nothing here is
clever, and nothing requires the model's author to have heard of xevals.

Two rules keep it honest.

**Every framework is imported inside the class that needs it.** ``import
xevals.adapters`` costs nothing on a numpy-only install, and a torch model raises
:class:`~xevals.errors.MissingExtra` at *wrap* time -- when the user can act on
it -- rather than at import time, when they cannot.

**NumPy at the boundary, in both directions.** A torch adapter takes numpy in,
makes a tensor, runs under ``no_grad``, and returns numpy. The core never sees a
tensor, which is what allows the core to depend on numpy alone.

Detection order matters, and it is deliberately conservative: an object that
*already* satisfies a protocol is returned untouched, because the model's own
implementation is by definition more faithful than anything inferred about it.
Only then does :func:`detect` look at base classes, and only then at
callability.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from xevals.core.errors import CapabilityMissing, MissingExtra
from xevals.core.registry import Registry
from xevals.core.types import Action, Obs, Plan, Planner, Policy, Prediction, Scorer, WorldModel

__all__ = [
    "ADAPTERS",
    "CallablePolicy",
    "ChatPlanner",
    "HFVLAPolicy",
    "JaxPolicy",
    "LeRobotPolicy",
    "RemotePolicy",
    "TorchPolicy",
    "XWMWorldModel",
    "available",
    "create",
    "describe",
    "detect",
    "wrap",
]

#: Named adapters. ``xevals adapters list`` reads this.
ADAPTERS: Registry[Any] = Registry("adapters")


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _features(obs: Obs, keys: tuple[str, ...] | None = None) -> np.ndarray:
    """Flatten an observation into one ``(D,)`` float32 vector.

    The fallback for models that want a plain vector and do not say which fields
    they read. Keys are visited in sorted order so the layout is stable across
    processes -- an unstable feature order is the kind of bug that shows up as
    "the model is fine locally and terrible in CI".
    """
    chosen = keys if keys is not None else tuple(sorted(k for k in obs if k != "instruction"))
    parts: list[np.ndarray] = []
    for key in chosen:
        value = obs.get(key)
        if value is None or isinstance(value, str):
            if keys is not None:
                raise ValueError(f"required numeric feature {key!r} is missing or nonnumeric")
            continue
        parts.append(np.asarray(value, dtype=np.float32).reshape(-1))
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)


def _to_numpy(value: Any) -> np.ndarray:
    """Host float32, including Torch dtypes NumPy cannot represent (bfloat16)."""
    for attr in ("detach", "cpu"):
        candidate = getattr(value, attr, None)
        if callable(candidate):
            value = candidate()
    if callable(getattr(value, "numpy", None)):
        if callable(getattr(value, "float", None)):
            value = value.float()
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


def validate_action(value: Any, action_dim: int | None = None) -> Action:
    """Validate one action without turning a batch or action chunk into a vector."""
    action = _to_numpy(value)
    if action.ndim == 2 and action.shape[0] == 1:
        action = action[0]
    if action.ndim != 1 or not action.size:
        raise ValueError(f"expected one action (A,) or (1, A), got {action.shape}")
    if action_dim is not None and action.shape != (action_dim,):
        raise ValueError(f"expected {action_dim} action entries, got {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("action contains NaN or infinity")
    return action.copy()


def validate_batch(value: Any, size: int, action_dim: int | None = None) -> np.ndarray:
    """Validate the batch axes; individual rows are validated by the runner."""
    actions = _to_numpy(value)
    if actions.ndim != 2 or actions.shape[0] != size or actions.shape[1] == 0:
        raise ValueError(f"expected actions (B, A) with B={size}, got {actions.shape}")
    if action_dim is not None and actions.shape[1] != action_dim:
        raise ValueError(f"expected {action_dim} action entries, got {actions.shape}")
    return actions.copy()


def _placement(module: Any, torch: Any, device: str | None) -> tuple[str, Any]:
    """Preserve a module's placement; parameterless modules also have buffers."""
    from itertools import chain

    tensors = list(chain(module.parameters(), module.buffers()))
    devices = {str(t.device) for t in tensors}
    if device is None and len(devices) > 1:
        raise ValueError("model spans multiple devices; supply an explicit device")
    chosen = str(device or next(iter(devices), "cpu"))
    dtype = next((t.dtype for t in tensors if t.is_floating_point()), torch.float32)
    return chosen, dtype


def _tensor(value: Any, torch: Any, device: str, dtype: Any) -> Any:
    tensor = torch.as_tensor(np.asarray(value))
    return tensor.to(device=device, dtype=dtype if tensor.is_floating_point() else tensor.dtype)


class _Described:
    """Mixin giving every adapter the same ``describe()`` shape.

    That shape lands verbatim in ``run.json`` as the model fingerprint, so two
    runs can be compared on *what was evaluated* and not only on the numbers.
    """

    adapter_name = "adapter"

    def describe(self) -> dict[str, Any]:
        """Framework, parameter count and placement, as far as they are knowable."""
        return {"adapter": self.adapter_name, **self._describe()}

    def _describe(self) -> dict[str, Any]:
        return {}


# --------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------


class CallablePolicy(_Described):
    """A plain function as a policy. The smallest thing that can be evaluated.

    Accepts three shapes, tried in order: ``f(obs, instruction=...)``,
    ``f(obs)``, and ``f(vector)`` for a function that wants a flat array. The
    inspection happens once, at construction, rather than per step -- a
    ``TypeError`` caught per call would swallow real errors raised *inside* the
    function and turn a bug into a silently different calling convention.
    """

    adapter_name = "callable"

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        action_dim: int | None = None,
        feature_keys: tuple[str, ...] | None = None,
        batch_fn: Callable[..., Any] | None = None,
    ) -> None:
        import inspect

        self.fn = fn
        self.batch_fn = batch_fn
        self.batch_mode = "stateless" if batch_fn is not None else None
        self.action_dim = action_dim
        self.feature_keys = feature_keys
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):  # builtins and C callables
            params = {}
        self._takes_instruction = "instruction" in params
        self._takes_obs = bool(params) and next(iter(params)) in ("obs", "observation", "o")

    def act(self, obs: Obs, *, instruction: str | None = None) -> Action:
        """One action, in whatever shape the wrapped function wants its input."""
        payload = obs if self._takes_obs else _features(obs, self.feature_keys)
        if self._takes_instruction:
            out = self.fn(payload, instruction=instruction)
        else:
            out = self.fn(payload)
        return validate_action(out, self.action_dim)

    def act_batch(self, observations: Sequence[Obs], *, instructions=None) -> np.ndarray:
        """Call an explicitly supplied stateless ``batch_fn(obs_list, instructions=...)``."""
        if self.batch_fn is None:
            raise CapabilityMissing("act_batch")
        return validate_batch(
            self.batch_fn(observations, instructions=instructions),
            len(observations), self.action_dim,
        )

    def _describe(self) -> dict[str, Any]:
        return {
            "callable": getattr(self.fn, "__name__", type(self.fn).__name__),
            "feature_keys": self.feature_keys,
            "action_dim": self.action_dim,
            "batch_mode": self.batch_mode,
        }


class TorchPolicy(_Described):
    """A ``torch.nn.Module`` as a policy.

    Does the four things every hand-written torch evaluation script does and
    usually gets subtly wrong: ``eval()`` mode, ``no_grad``, the device round
    trip, and a batch dimension added on the way in and removed on the way out.
    A forgotten ``eval()`` leaves dropout on and costs a few points of success
    rate that then get attributed to the method.
    """

    adapter_name = "torch"

    def __init__(
        self,
        module: Any,
        *,
        device: str | None = None,
        feature_keys: tuple[str, ...] | None = None,
        dict_input: bool | None = None,
        action_dim: int | None = None,
        batch_mode: str | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise MissingExtra("torch", "torch", "to wrap a torch.nn.Module") from exc
        self._torch = torch
        if batch_mode not in (None, "stateless"):
            raise ValueError("batch_mode must be None or 'stateless'")
        self.device, self.dtype = _placement(module, torch, device)
        self.module = module.to(self.device).eval()
        self.feature_keys = feature_keys
        self.action_dim = action_dim
        self.batch_mode = batch_mode
        # A module whose forward names an observation dict takes one; otherwise
        # it gets the flattened feature vector.
        self.dict_input = (
            dict_input
            if dict_input is not None
            else "obs" in getattr(module.forward, "__code__", type("", (), {"co_varnames": ()}))
            .co_varnames[:2]
        )

    def __getattr__(self, name: str) -> Any:
        # Preserve recurrent/chunked module state management on the serial path,
        # without advertising reset on modules that do not implement it.
        if name == "reset":
            return getattr(self.module, name)
        raise AttributeError(name)

    def _payload(self, observations: Sequence[Obs]) -> Any:
        torch = self._torch
        if not self.dict_input:
            values = np.stack([_features(obs, self.feature_keys) for obs in observations])
            return _tensor(values, torch, self.device, self.dtype)
        keys = self.feature_keys or tuple(
            k for k, v in observations[0].items() if v is not None and not isinstance(v, str)
        )
        if any(set(k for k, v in obs.items() if v is not None and not isinstance(v, str))
               != set(keys) for obs in observations) and self.feature_keys is None:
            raise ValueError("batched dict observations must have identical numeric keys")
        return {
            k: _tensor(np.stack([obs[k] for obs in observations]), torch, self.device, self.dtype)
            for k in keys
        }

    def act(self, obs: Obs, *, instruction: str | None = None) -> Action:
        """One action, with floating inputs matching the checkpoint's dtype."""
        with self._torch.no_grad():
            out = self.module(self._payload([obs]))
        return validate_action(out, self.action_dim)

    def act_batch(self, observations: Sequence[Obs], *, instructions=None) -> np.ndarray:
        """Independent rows; opt in only for a stateless, batch-independent module."""
        if self.batch_mode != "stateless":
            raise CapabilityMissing("stateless act_batch")
        with self._torch.no_grad():
            out = self.module(self._payload(observations))
        return validate_batch(out, len(observations), self.action_dim)

    def encode(self, obs: Obs) -> np.ndarray:
        """The module's ``encode``, when it has one. Enables representation metrics."""
        encode = getattr(self.module, "encode", None)
        if encode is None:
            from xevals.core.errors import CapabilityMissing

            raise CapabilityMissing("encode")
        with self._torch.no_grad():
            return _to_numpy(encode(self._payload([obs]))).reshape(-1)

    def cost(self) -> dict[str, Any]:
        """Parameter count, device and dtype -- the efficiency dimension's inputs."""
        params = sum(int(p.numel()) for p in self.module.parameters())
        dtypes = {str(p.dtype) for p in self.module.parameters()}
        return {"params": params, "device": str(self.device), "dtype": sorted(dtypes)}

    def _describe(self) -> dict[str, Any]:
        return {
            "module": type(self.module).__name__, "feature_keys": self.feature_keys,
            "dict_input": self.dict_input, "batch_mode": self.batch_mode,
            "action_dim": self.action_dim, **self.cost(),
        }


class JaxPolicy(_Described):
    """A JAX or Equinox callable as a policy.

    ``jax.device_get`` on the way out, so the core never holds a device array
    and never has to know whether JAX is installed.
    """

    adapter_name = "jax"

    def __init__(self, fn: Any, *, feature_keys: tuple[str, ...] | None = None) -> None:
        try:
            import jax
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise MissingExtra("jax", "jax", "to wrap a JAX callable") from exc
        self._jax = jax
        self.fn = fn
        self.feature_keys = feature_keys

    def act(self, obs: Obs, *, instruction: str | None = None) -> Action:
        """One action, pulled back to the host."""
        out = self.fn(self._jax.numpy.asarray(_features(obs, self.feature_keys)))
        return validate_action(self._jax.device_get(out))

    def _describe(self) -> dict[str, Any]:
        return {"callable": getattr(self.fn, "__name__", type(self.fn).__name__)}


class LeRobotPolicy(_Described):
    """A LeRobot policy, driven through its own ``select_action`` loop.

    LeRobot policies keep an action queue for chunked prediction and a
    normalisation module fitted to the training dataset. Both are the policy's
    business, so this adapter calls ``select_action`` and ``reset`` rather than
    reimplementing either -- reimplementing normalisation is the single most
    common way a wrapped VLA quietly loses most of its performance.
    """

    adapter_name = "lerobot"

    def __init__(self, policy: Any, *, device: str | None = None) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise MissingExtra("torch", "lerobot", "to wrap a LeRobot policy") from exc
        self._torch = torch
        self.device, self.dtype = _placement(policy, torch, device)
        self.policy = policy.to(self.device).eval()

    def reset(self) -> None:
        """Clear the action queue between episodes. Skipping this leaks state."""
        reset = getattr(self.policy, "reset", None)
        if reset is not None:
            reset()

    def act(self, obs: Obs, *, instruction: str | None = None) -> Action:
        """One action from the policy's own chunked loop."""
        torch = self._torch
        batch = {}
        for key, value in obs.items():
            if isinstance(value, str):
                continue
            tensor = torch.as_tensor(np.asarray(value)).to(self.device).unsqueeze(0)
            if key == "image" and tensor.ndim == 4 and tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(0, 3, 1, 2).float() / 255.0
                batch["observation.image"] = tensor
            elif key == "state":
                batch["observation.state"] = tensor.float()
            else:
                batch[f"observation.{key}"] = tensor
        if instruction is not None:
            batch["task"] = [instruction]
        with torch.no_grad():
            action = self.policy.select_action(batch)
        return validate_action(action)

    def _describe(self) -> dict[str, Any]:
        params = sum(int(p.numel()) for p in self.policy.parameters())
        return {"policy": type(self.policy).__name__, "params": params, "device": self.device}


class HFVLAPolicy(_Described):
    """An OpenVLA-style vision-language-action model from ``transformers``.

    The two things that make a VLA hard to evaluate correctly are both handled
    here and both stated, because getting either wrong produces a plausible but
    wrong number:

    * **The prompt template is part of the checkpoint.** A model trained on
      ``"In: What action should the robot take to {instruction}?\\nOut:"`` and
      prompted with a bare instruction is out of distribution. The template is a
      constructor argument, not a constant.
    * **Actions come back normalised** against a per-dataset statistics blob. If
      the caller supplies ``unnorm_key``, the model's own un-normalisation is
      used; otherwise the raw output is returned and the run records that it was.
    """

    adapter_name = "hf_vla"

    def __init__(
        self,
        model_id: str,
        *,
        prompt: str = "In: What action should the robot take to {instruction}?\nOut:",
        unnorm_key: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise MissingExtra("transformers", "torch", f"to load the VLA {model_id!r}") from exc
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, dtype or ("float32" if self.device == "cpu" else "bfloat16"))
        self.prompt = prompt
        self.unnorm_key = unnorm_key
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=self.dtype, trust_remote_code=True
        ).to(self.device).eval()
        self.model_id = model_id

    def act(self, obs: Obs, *, instruction: str | None = None) -> Action:
        """One action, through the checkpoint's own prompt and un-normalisation."""
        from PIL import Image

        image = obs.get("image")
        if image is None:
            raise ValueError("a VLA needs an 'image' field in the observation")
        text = self.prompt.format(instruction=instruction or obs.get("instruction", ""))
        inputs = self.processor(
            text, Image.fromarray(np.asarray(image, dtype=np.uint8)), return_tensors="pt"
        )
        inputs = {
            k: v.to(device=self.device, dtype=self.dtype if v.is_floating_point() else v.dtype)
            for k, v in inputs.items()
        }
        with self._torch.no_grad():
            kwargs = {"unnorm_key": self.unnorm_key} if self.unnorm_key else {}
            action = self.model.predict_action(**inputs, do_sample=False, **kwargs)
        return validate_action(action)

    def _describe(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "device": self.device,
            "dtype": str(self.dtype),
            "prompt": self.prompt,
            "unnorm_key": self.unnorm_key,
            "normalised_output": self.unnorm_key is None,
        }


class RemotePolicy(_Described):
    """A policy behind an HTTP endpoint. The black-box case.

    Worth supporting first-class rather than as a workaround: a hosted model is
    exactly the kind that cannot be inspected, differentiated or profiled, so
    the dimensions xevals measures without inspection -- robustness, security,
    consistency, end-to-end latency -- are most of what can be known about it.

    Latency is measured client-side and includes the network, which is the
    honest number for a deployment that would also call it over the network.
    """

    adapter_name = "remote"

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
        action_key: str = "action",
    ) -> None:
        self.url = url
        self.timeout = timeout
        self.headers = {"content-type": "application/json", **(headers or {})}
        self.action_key = action_key
        self.last_latency_ms = 0.0

    def act(self, obs: Obs, *, instruction: str | None = None) -> Action:
        """POST the observation as JSON, read the action back."""
        import json
        import urllib.request

        payload = {
            "observation": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in obs.items()
                if not isinstance(v, np.ndarray) or v.size <= 1_000_000
            },
            "instruction": instruction,
        }
        request = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(), headers=self.headers
        )
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
            body = json.loads(response.read().decode())
        self.last_latency_ms = (time.perf_counter() - start) * 1e3
        return validate_action(body[self.action_key])

    def _describe(self) -> dict[str, Any]:
        return {"url": self.url, "timeout_s": self.timeout}


class XWMWorldModel(_Described):
    """A world model from the sibling library, behind xevals's protocol.

    xwm's models are Equinox PyTrees operating on latents; the bridge is thin
    because both libraries agreed on numpy at the boundary. Where the xwm model
    exposes an encoder, ``encode`` is forwarded, which is what lets the
    prediction metrics work in latent space rather than only in pixels.
    """

    adapter_name = "xwm"

    def __init__(self, model: Any) -> None:
        self.model = model

    def predict(self, obs: Obs, actions: np.ndarray, *, horizon: int = 1) -> Prediction:
        """Roll ``actions`` forward, returning whatever the model actually predicts."""
        import jax.numpy as jnp

        latent = self.encode(obs)
        rollout = getattr(self.model, "rollout", None) or self.model.imagine
        out = rollout(jnp.asarray(latent), jnp.asarray(actions[:horizon], dtype=jnp.float32))
        if isinstance(out, dict):
            return {k: np.asarray(v) for k, v in out.items()}
        return {"latents": np.asarray(out)}

    def encode(self, obs: Obs) -> np.ndarray:
        """The model's own encoder, so latents mean what the model means by them."""
        import jax.numpy as jnp

        encode = getattr(self.model, "encode", None)
        if encode is None:
            from xevals.core.errors import CapabilityMissing

            raise CapabilityMissing("encode")
        image = obs.get("image")
        payload = _features(obs) if image is None else np.asarray(image, dtype=np.float32) / 255.0
        return np.asarray(encode(jnp.asarray(payload))).reshape(-1)

    def _describe(self) -> dict[str, Any]:
        return {"model": type(self.model).__name__}


class ChatPlanner(_Described):
    """An LLM as a task planner. Anything ``str -> str`` will do.

    Takes a bare callable, an Anthropic client, or an OpenAI-compatible client,
    because pinning one provider would make the planner dimensions unmeasurable
    for everyone using the other. Temperature is 0 and the seed is passed where
    the API accepts one: a consistency metric on a sampled model measures the
    sampler.

    Refusal is detected by :mod:`xevals.judges`, not here, so the rule is visible
    and replaceable rather than buried in an adapter.
    """

    adapter_name = "chat"

    def __init__(
        self,
        client: Any,
        *,
        model: str | None = None,
        system: str = "You are a robot task planner. Answer with numbered steps.",
        max_tokens: int = 512,
    ) -> None:
        self.client = client
        self.model = model
        self.system = system
        self.max_tokens = max_tokens
        self._kind = _chat_kind(client)

    def plan(self, instruction: str, *, obs: Obs | None = None) -> Plan:
        """Ask for a plan and return it with its steps split out."""
        text = self._complete(instruction)
        steps = [
            line.strip(" -*0123456789.")
            for line in text.splitlines()
            if line.strip() and line.strip()[0] in "-*0123456789"
        ]
        return {"text": text, "steps": steps}

    def _complete(self, instruction: str) -> str:
        if self._kind == "callable":
            return str(self.client(instruction))
        if self._kind == "anthropic":
            message = self.client.messages.create(
                model=self.model or "claude-sonnet-5",
                max_tokens=self.max_tokens,
                temperature=0.0,
                system=self.system,
                messages=[{"role": "user", "content": instruction}],
            )
            return "".join(block.text for block in message.content if hasattr(block, "text"))
        completion = self.client.chat.completions.create(
            model=self.model or "gpt-4o-mini",
            temperature=0.0,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": instruction},
            ],
        )
        return completion.choices[0].message.content or ""

    def _describe(self) -> dict[str, Any]:
        return {"client": self._kind, "model": self.model, "temperature": 0.0}


def _chat_kind(client: Any) -> str:
    """Which of the three chat shapes this client is."""
    if hasattr(client, "messages") and hasattr(client.messages, "create"):
        return "anthropic"
    if hasattr(client, "chat"):
        return "openai"
    if callable(client):
        return "callable"
    raise TypeError(
        "a chat client must be callable, or expose .messages.create (Anthropic) "
        "or .chat.completions.create (OpenAI-compatible)"
    )


# --------------------------------------------------------------------------
# Detection and the public entry point
# --------------------------------------------------------------------------


def detect(obj: Any) -> str | None:
    """Which adapter fits ``obj``, or ``None`` if it already satisfies a protocol.

    Checks base classes by *name* rather than by importing the framework, so
    detection never pulls torch into a process that does not have it.
    """
    if isinstance(obj, (Policy, WorldModel, Planner, Scorer)):
        return None
    bases = {c.__module__.split(".")[0] + "." + c.__name__ for c in type(obj).__mro__}
    if any("lerobot" in c.__module__ for c in type(obj).__mro__):
        return "lerobot"
    if any(c.__module__.split(".")[0] == "xwm" for c in type(obj).__mro__):
        return "xwm"
    if any(b == "torch.Module" for b in bases):
        return "torch"
    if any(c.__module__.split(".")[0] in ("jax", "equinox") for c in type(obj).__mro__):
        return "jax"
    if hasattr(obj, "messages") or hasattr(obj, "chat"):
        return "chat"
    if isinstance(obj, str) and obj.startswith(("http://", "https://")):
        return "remote"
    if callable(obj):
        return "callable"
    return None


def wrap(obj: Any, *, kind: str | None = None, adapter: str | None = None, **hints: Any) -> Any:
    """Turn any model into something xevals can evaluate.

    Args:
        obj: the model. A callable, a torch module, a JAX function, a LeRobot or
            HuggingFace policy, an xwm world model, a chat client, or a URL.
        kind: force the model kind -- ``"policy"``, ``"world_model"``,
            ``"planner"``, ``"scorer"``. Selects the interface for a bare callable;
            for other models, validates that the adapter implements this kind.
        adapter: force the adapter by name, skipping detection.
        **hints: passed to the adapter's constructor (``device``, ``prompt``,
            ``unnorm_key``, ``feature_keys``, ...).

    Returns:
        The object itself when it already satisfies a protocol, otherwise an
        adapter around it.

    Raises:
        TypeError: when nothing fits, listing what was tried. An unwrappable
            model is a real answer, and a confusing one is worse than a clear
            refusal.

    Examples:
        >>> import numpy as np, xevals
        >>> policy = xevals.wrap(lambda obs: np.zeros(2, dtype=np.float32))
        >>> policy.act({"state": np.zeros(4, dtype=np.float32)}).shape
        (2,)
    """
    kinds = {"policy": "act", "world_model": "predict", "planner": "plan", "scorer": "score"}
    if kind is not None and kind not in kinds:
        raise ValueError(f"unknown model kind {kind!r}")
    name = adapter or detect(obj)
    if name is None:
        if not isinstance(obj, (Policy, WorldModel, Planner, Scorer)):
            raise TypeError(f"cannot wrap {type(obj).__name__}; supply a model protocol or adapter")
        result = obj
    elif name == "callable" and kind and kind != "policy":
        result = _CallableOther(obj, kind)
    else:
        result = ADAPTERS.create(name, model=obj, **hints)
    if kind is not None and not callable(getattr(result, kinds[kind], None)):
        raise TypeError(f"adapter {name!r} does not implement model kind {kind!r}")
    return result


class _CallableOther(_Described):
    """A bare callable declared to be a world model, planner or scorer.

    Separate from :class:`CallablePolicy` because the *method name* is the whole
    contract: a function called ``predict`` and a function called ``act`` are
    evaluated by different metrics, and a signature cannot say which one it is.
    """

    adapter_name = "callable"

    def __init__(self, fn: Callable[..., Any], kind: str) -> None:
        self.fn = fn
        self.kind = kind
        if kind == "world_model":
            self.predict = self._predict  # type: ignore[assignment]
        elif kind == "planner":
            self.plan = self._plan  # type: ignore[assignment]
        elif kind == "scorer":
            self.score = self._score  # type: ignore[assignment]
        else:
            raise ValueError(f"unknown model kind {kind!r}")

    def _predict(self, obs: Obs, actions: np.ndarray, *, horizon: int = 1) -> Prediction:
        out = self.fn(obs, actions[:horizon])
        return out if isinstance(out, dict) else {"latents": np.asarray(out)}

    def _plan(self, instruction: str, *, obs: Obs | None = None) -> Plan:
        out = self.fn(instruction)
        return out if isinstance(out, dict) else {"text": str(out)}

    def _score(self, obs: Obs, *, instruction: str | None = None) -> float:
        return float(self.fn(obs))

    def _describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "callable": getattr(self.fn, "__name__", "<lambda>")}


ADAPTERS.register(
    "callable",
    lambda model, **kw: CallablePolicy(model, **kw),
    summary="a plain function as a policy",
)
ADAPTERS.register(
    "torch",
    lambda model, **kw: TorchPolicy(model, **kw),
    summary="a torch.nn.Module, in eval mode, under no_grad",
    requires=("torch",),
)
ADAPTERS.register(
    "jax",
    lambda model, **kw: JaxPolicy(model, **kw),
    summary="a JAX or Equinox callable",
    requires=("jax",),
)
ADAPTERS.register(
    "lerobot",
    lambda model, **kw: LeRobotPolicy(model, **kw),
    summary="a LeRobot policy, through its own select_action loop",
    requires=("lerobot",),
)
ADAPTERS.register(
    "hf_vla",
    lambda model, **kw: HFVLAPolicy(model, **kw),
    summary="an OpenVLA-style VLA from transformers",
    requires=("torch",),
)
ADAPTERS.register(
    "remote",
    lambda model, **kw: RemotePolicy(model, **kw),
    summary="a policy behind an HTTP endpoint",
)
ADAPTERS.register(
    "xwm",
    lambda model, **kw: XWMWorldModel(model, **kw),
    summary="a world model from the sibling xwm library",
    requires=("xwm",),
)
ADAPTERS.register(
    "chat",
    lambda model, **kw: ChatPlanner(model, **kw),
    summary="an LLM as a task planner",
)


def available() -> list[str]:
    """Registered adapter names."""
    return ADAPTERS.available()


def describe(name: str | None = None) -> dict[str, Any]:
    """What an adapter wraps, without importing its framework."""
    return ADAPTERS.describe(name)  # type: ignore[return-value]


def create(name: str, model: Any, **kwargs: Any) -> Any:
    """Build a named adapter around ``model``."""
    return ADAPTERS.create(name, model=model, **kwargs)


def fingerprint(model: Any) -> dict[str, Any]:
    """What was evaluated, as far as the model will say.

    Goes into ``run.json`` verbatim. A run whose model cannot be identified is
    not reproducible, so this reaches for ``describe``, then ``cost``, then the
    type name, and records which of them it got.
    """
    out: dict[str, Any] = {"type": type(model).__name__}
    describe_fn = getattr(model, "describe", None)
    if callable(describe_fn):
        try:
            out.update(describe_fn())
        except Exception as exc:  # pragma: no cover - a model's own describe failing
            out["describe_error"] = str(exc)
    cost = getattr(model, "cost", None)
    if callable(cost):
        try:
            out.update(cost())
        except Exception:  # pragma: no cover
            pass
    out["capabilities"] = sorted(
        name
        for name, attr in (
            ("act", "act"),
            ("predict", "predict"),
            ("plan", "plan"),
            ("score", "score"),
            ("confidence", "confidence"),
            ("encode", "encode"),
            ("reset", "reset"),
        )
        if callable(getattr(model, attr, None))
    )
    if getattr(model, "batch_mode", None) == "stateless" and callable(
        getattr(model, "act_batch", None)
    ):
        out["capabilities"].append("act_batch")
        out["capabilities"].sort()
        out["batch_mode"] = "stateless"
    return out
