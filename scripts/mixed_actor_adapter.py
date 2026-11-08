"""Expand the verified flat actor from 67 to 95 inputs without changing its mean."""

from __future__ import annotations

import torch
from torch.nn import functional as F


SOURCE_OBSERVATIONS = 67
MIXED_OBSERVATIONS = 95
EXTRA_OBSERVATIONS = 28
NORMALIZATION_EPSILON = .01
SOURCE_SHA256 = "725c9696e82ef25e4b13e20c648562d17be6c79a6c1bc6ebd81936fbf2590527"
JOINT_NAMES = [f"L{leg}_{part}" for leg in range(1, 7) for part in ("coxa", "femur", "tibia")]
PREFIX_TERMS = ["joint_pos", "joint_vel", "base_lin_vel", "base_ang_vel", "projected_gravity",
                "command", "height_command", "last_action"]
STD_KEY = "distribution.log_std_param"


def adapt_flat_actor(source):
    """Copy the source; zero only new input columns and initialize new moments."""
    expected = {"mlp.0.weight": (256, 67), "mlp.0.bias": (256,),
                "mlp.2.weight": (256, 256), "mlp.2.bias": (256,),
                "mlp.4.weight": (128, 256), "mlp.4.bias": (128,),
                "mlp.6.weight": (18, 128), "mlp.6.bias": (18,), STD_KEY: (18,),
                "obs_normalizer._mean": (1, 67), "obs_normalizer._var": (1, 67),
                "obs_normalizer._std": (1, 67), "obs_normalizer.count": ()}
    if set(source) != set(expected):
        raise ValueError("Source actor must match the complete 67-input ELU/Gaussian contract")
    for key, shape in expected.items():
        value = source[key]
        dtype = torch.int64 if key == "obs_normalizer.count" else torch.float32
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != dtype
                or not torch.isfinite(value).all().item()):
            raise ValueError(f"Invalid source actor tensor: {key}")
    if (source["obs_normalizer._var"] < 0).any() or (source["obs_normalizer._std"] < 0).any() or source["obs_normalizer.count"] < 0:
        raise ValueError("Invalid source normalizer moments/count")
    adapted = {key: value.detach().clone() for key, value in source.items()}
    weight = source["mlp.0.weight"]
    adapted["mlp.0.weight"] = torch.cat((weight, weight.new_zeros(256, EXTRA_OBSERVATIONS)), dim=1)
    for key, initial in (("obs_normalizer._mean", 0.), ("obs_normalizer._var", 1.), ("obs_normalizer._std", 1.)):
        value = source[key]
        adapted[key] = torch.cat((value, value.new_full((1, EXTRA_OBSERVATIONS), initial)), dim=1)
    return adapted


@torch.no_grad()
def mean_output(state, observations):
    value = (observations - state["obs_normalizer._mean"]) / (state["obs_normalizer._std"] + NORMALIZATION_EPSILON)
    for layer in (0, 2, 4, 6):
        value = F.linear(value, state[f"mlp.{layer}.weight"], state[f"mlp.{layer}.bias"])
        if layer != 6:
            value = F.elu(value)
    return value


def frozen_input_equivalence(source, adapted, physical_prefix=None):
    """CPU numerical check with matched raw prefixes and varied arbitrary extras."""
    source = {key: value.detach().cpu() for key, value in source.items()}
    adapted = {key: value.detach().cpu() for key, value in adapted.items()}
    if tuple(adapted["mlp.0.weight"].shape) != (256, 95) or torch.count_nonzero(adapted["mlp.0.weight"][:, 67:]):
        raise ValueError("New actor input columns must be exactly zero at initialization")
    for key, value in source.items():
        if key == STD_KEY:
            continue  # Exploration is deliberately reinitialized independently.
        actual = adapted[key]
        if key in ("mlp.0.weight", "obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"):
            actual = actual[..., :67]
        if not torch.equal(actual, value):
            raise ValueError(f"Original actor prefix changed: {key}")
    for key, expected in (("obs_normalizer._mean", 0.), ("obs_normalizer._var", 1.), ("obs_normalizer._std", 1.)):
        if not torch.equal(adapted[key][..., 67:], torch.full((1, 28), expected)):
            raise ValueError(f"New normalizer feature initialization differs: {key}")
    generator = torch.Generator().manual_seed(1973)
    normalized = torch.randn(256, 67, generator=generator)
    prefixes = source["obs_normalizer._mean"] + normalized * (source["obs_normalizer._std"] + NORMALIZATION_EPSILON)
    physical_count = 0
    if physical_prefix is not None:
        physical_prefix = physical_prefix.detach().cpu().to(torch.float32)
        if physical_prefix.ndim != 2 or physical_prefix.shape[1] != 67 or not torch.isfinite(physical_prefix).all():
            raise ValueError("Physical policy prefix must be finite N x 67")
        physical_count = len(physical_prefix)
        prefixes = torch.cat((prefixes, physical_prefix), dim=0)
    old_mean = mean_output(source, prefixes)
    maximum_error = 0.
    for scale in (0., 1., 10.):
        extras = torch.randn(len(prefixes), 28, generator=generator) * scale
        new_mean = mean_output(adapted, torch.cat((prefixes, extras), dim=1))
        maximum_error = max(maximum_error, float((new_mean - old_mean).abs().max()))
        if not torch.allclose(new_mean, old_mean, rtol=2e-5, atol=5e-5):
            raise ValueError(f"Frozen-input actor equivalence failed: maximum absolute error {maximum_error}")
    return {"original_input_count": 67, "expanded_input_count": 95, "new_columns_exactly_zero": True,
            "original_actor_and_normalizer_prefix_exact": True, "normalization_epsilon": .01,
            "normalizer_count_retained": int(adapted["obs_normalizer.count"].item()),
            "new_normalizer_mean": 0., "new_normalizer_var": 1., "new_normalizer_std": 1.,
            "probe_prefixes": len(prefixes), "physical_prefixes": physical_count,
            "extra_feature_scales": [0., 1., 10.], "max_abs_action_error": maximum_error,
            "tolerance": {"atol": 5e-5, "rtol": 2e-5},
            "scope": "Equality at initialization for the same raw 67-input prefix; subsequent PPO and normalization updates may change behavior."}


def initialize_mixed_runner(runner, checkpoint, physical_prefix=None, expected_std=.05):
    from train_flat_wsp import actor_only_warmstart

    if int(runner.alg.critic.state_dict()["obs_normalizer.count"].item()) != 0:
        raise ValueError("Mixed warmstart requires a fresh critic normalizer")
    source = checkpoint["actor_state_dict"]
    adapted = adapt_flat_actor(source)
    proof = frozen_input_equivalence(source, adapted, physical_prefix)
    expanded_checkpoint = {**checkpoint, "actor_state_dict": adapted}
    loading = actor_only_warmstart(runner, expanded_checkpoint, expected_std=expected_std)
    loaded = runner.alg.get_policy().state_dict()
    proof["loaded_actor_equivalence"] = frozen_input_equivalence(source, loaded, physical_prefix)
    proof["actor_only_load"] = loading
    proof["normalizer_update_note"] = "The scalar source count is retained. All 95 feature moments subsequently follow the installed normalizer update rule; the new moments are initial values, not a frozen normalizer."
    return proof
