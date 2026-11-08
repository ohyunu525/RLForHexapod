"""Create an experimental flat-policy saturation-release checkpoint.

This is a closed-form final-layer weight edit, not RL or a gait controller.
Synthetic probes describe network sensitivity, never observed locomotion.
The original checkpoint is preserved; physics evaluation is still required.
Default scope is negative tibia saturation. Explicit --joint-scope all also
allows both signs of coxa/femur saturation; it does not widen tibia direction.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


SOURCE_SHA256 = "631f6af6f01b5b836c8c83fa7a25d74ce985a112226d217a1938f1c277f38493"
JOINT_NAMES = [f"L{leg}_{part}" for leg in range(1, 7) for part in ("coxa", "femur", "tibia")]
TIBIA_ROWS = [2, 5, 8, 11, 14, 17]
HEAD_WEIGHT, HEAD_BIAS = "mlp.6.weight", "mlp.6.bias"
# Verified from this run's exported TorchScript obs_normalizer.eps, not a generic default.
EPSILON = .01


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


class FlatActor(nn.Module):
    """Exact known flat actor; accepts RAW policy observations, including batches."""

    def __init__(self, state_dict):
        super().__init__()
        shapes = {
            "obs_normalizer._mean": (1, 67), "obs_normalizer._std": (1, 67),
            "mlp.0.weight": (256, 67), "mlp.0.bias": (256,),
            "mlp.2.weight": (256, 256), "mlp.2.bias": (256,),
            "mlp.4.weight": (128, 256), "mlp.4.bias": (128,),
            HEAD_WEIGHT: (18, 128), HEAD_BIAS: (18,),
        }
        for key, shape in shapes.items():
            tensor = state_dict.get(key)
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
                raise ValueError(f"Unsupported flat-policy tensor: {key}; expected {shape}")
            if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                raise ValueError(f"Expected finite float32 values: {key}")
        if (state_dict["obs_normalizer._std"] < 0).any():
            raise ValueError("Normalizer standard deviations must be nonnegative")
        self.register_buffer("mean", state_dict["obs_normalizer._mean"].detach().clone())
        self.register_buffer("std", state_dict["obs_normalizer._std"].detach().clone())
        self.mlp = nn.Sequential(nn.Linear(67, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(),
                                 nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 18))
        self.mlp.load_state_dict({key[4:]: value for key, value in state_dict.items()
                                 if key.startswith("mlp.")}, strict=True)
        self.eval()

    def features(self, observations):
        if observations.ndim < 1 or observations.shape[-1] != 67:
            raise ValueError("Expected raw observations with last dimension 67")
        return self.mlp[:6]((observations - self.mean) / (self.std + EPSILON))

    def forward(self, observations):
        return self.mlp[6](self.features(observations))


def validate_observations(observations):
    if observations.ndim != 2 or observations.shape[1] != 67 or observations.shape[0] < 2:
        raise ValueError("Need at least two raw 67-dimensional observations")
    if not observations.is_floating_point() or not torch.isfinite(observations).all():
        raise ValueError("Observations must contain only finite floating-point values")


def negative_tail_target(actions, strength):
    """C1 tail compression, identity above the knee, interior asymptote below it.

    For strength .2 the knee is -.8 and the asymptote is -.9. Its slope
    matches the identity at -.8. Thus an originally clipped negative response
    has a target inside [-1, 1], without flipping the response's sign.
    """
    if not math.isfinite(strength) or not 0 < strength <= .4:
        raise ValueError("strength must be finite and in (0, .4]")
    knee, width = 1.0 - strength, strength / 2.0
    softened = -knee - width * torch.tanh((-actions - knee) / width)
    return torch.where(actions < -knee, softened, actions)


def forward_mask(observations):
    # q18, qdot18, v3, omega3, gravity3, command3, height1, previous action18.
    command = observations[:, 45:48]
    return ((command[:, 0] >= .05) & (command[:, 1].abs() <= .025)
            & (command[:, 2].abs() <= .05))


def stats(values):
    values = values.detach().double().reshape(-1)
    if not values.numel():
        return {"count": 0}
    return {"count": values.numel(), "mean": values.mean().item(),
            "rms": values.square().mean().sqrt().item(), "max_abs": values.abs().max().item()}


@torch.inference_mode()
def synthetic_probes(state_dict, seed=13250, samples_per_case=256):
    """Deterministic moment-centered probes with constrained raw coordinates.

    These have no observed state covariance/contact labels and are NOT sampled
    simulation trajectories. Exact anchor rows include reset and stored mean.
    """
    if type(samples_per_case) is not int or samples_per_case < 2:
        raise ValueError("samples_per_case must be at least two")
    actor = FlatActor(state_dict)
    generator = torch.Generator().manual_seed(seed)
    mean, std = actor.mean[0], actor.std[0]
    cases = [("forward_010", .1, 0.), ("forward_015", .15, 0.), ("forward_020", .2, 0.),
             ("stand", 0., 0.), ("turn_left", 0., .25), ("turn_right", 0., -.25)]
    blocks, labels = [], []
    scale = torch.tensor([.5, 1., 1.] * 6)
    for label, vx, yaw in cases:
        obs = mean.expand(samples_per_case, -1).clone()
        jitter = torch.randn(samples_per_case, 67, generator=generator).clamp(-2, 2)
        obs[:, :18] += .2 * std[:18] * jitter[:, :18]
        q = obs[:, :18].reshape(-1, 6, 3)
        q[:, :, 0].clamp_(-.6, .6)
        q[:, :, 1].clamp_(-math.radians(20), math.radians(160))
        q[:, :, 2].clamp_(math.radians(-140), math.radians(40))
        obs[:, 18:36] = .15 * std[18:36] * jitter[:, 18:36]
        obs[:, 36:39] = .015 * jitter[:, 36:39]
        obs[:, 36] += vx
        obs[:, 39:42] = .03 * jitter[:, 39:42]
        obs[:, 41] += yaw
        gravity = torch.cat((.025 * jitter[:, 42:44], -torch.ones(samples_per_case, 1)), 1)
        obs[:, 42:45] = gravity / gravity.norm(dim=1, keepdim=True)
        obs[:, 45:48] = torch.tensor([vx, 0., yaw])
        obs[:, 48] = torch.linspace(.17, .24, samples_per_case)
        obs[:, 49:67] = (obs[:, :18] / scale).clamp(-1, 1)
        blocks.append(obs)
        labels.extend([label] * len(obs))
    reset = torch.zeros(1, 67)
    reset[:, 44], reset[:, 48] = -1., .205
    blocks.extend([reset, mean.unsqueeze(0)])
    labels.extend(["reset_anchor", "stored_moment_anchor"])
    observations = torch.cat(blocks)
    validate_observations(observations)
    return observations, {"kind": "synthetic_physical_range_and_moment_probes", "seed": seed,
        "cases": [case[0] for case in cases] + labels[-2:], "samples": len(observations),
        "samples_per_case": samples_per_case, "observed_trajectories": False,
        "limitations": ["Only marginal running moments are available; state covariance is unknown.",
                        "Joint bounds and normalized gravity do not establish physical realizability.",
                        "These probes cannot measure gait, contact, speed, or stability."]}


def load_trace(path, expected_sha256, max_samples=16384):
    """Read raw observations [sample, 67] or [time, env, 67], never pickle."""
    with np.load(path, allow_pickle=False) as archive:
        observation_key = "obs" if "obs" in archive else "observations"
        if observation_key not in archive:
            raise ValueError("Trace needs raw obs [sample, 67] or [time, env, 67]")
        observations = archive[observation_key]
        if observations.ndim not in (2, 3) or observations.shape[-1] != 67:
            raise ValueError("Trace observations must have shape [sample, 67] or [time, env, 67]")
        metadata = {}
        if "metadata" in archive:
            value = archive["metadata"].item()
            metadata = json.loads(value.decode() if isinstance(value, bytes) else str(value))
            if not isinstance(metadata, dict):
                raise ValueError("Trace metadata must be a JSON object")
        source_hash = metadata.get("source_sha256")
        if "source_sha256" in archive:
            value = archive["source_sha256"].item()
            explicit_hash = value.decode() if isinstance(value, bytes) else str(value)
            if source_hash is not None and source_hash != explicit_hash:
                raise ValueError("Trace source identities disagree")
            source_hash = explicit_hash
        if source_hash != expected_sha256:
            raise ValueError("Trace source_sha256 does not match the checkpoint")
        joint_names = metadata.get("joint_names")
        if "joint_names" in archive:
            explicit_names = archive["joint_names"].tolist()
            if joint_names is not None and joint_names != explicit_names:
                raise ValueError("Trace joint order metadata disagrees")
            joint_names = explicit_names
        if joint_names != JOINT_NAMES:
            raise ValueError("Trace joint order does not match the verified flat contract")
        epsilon = metadata.get("normalization_epsilon")
        if "normalization_epsilon" in archive:
            explicit_epsilon = float(archive["normalization_epsilon"].item())
            if epsilon is not None and not math.isclose(float(epsilon), explicit_epsilon, rel_tol=1e-6, abs_tol=1e-9):
                raise ValueError("Trace normalization epsilon metadata disagrees")
            epsilon = explicit_epsilon
        if epsilon is None or not math.isclose(float(epsilon), EPSILON, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError("Trace must identify normalization_epsilon=0.01")
        if observations.dtype.kind != "f" or not np.isfinite(observations).all():
            raise ValueError("Trace observations must be finite floating-point values")
        shape = list(observations.shape)
        flattened = observations.reshape(-1, 67)
        if len(flattened) > max_samples:
            indices = np.linspace(0, len(flattened) - 1, max_samples, dtype=np.int64)
            flattened = flattened[indices]
        result = torch.from_numpy(np.array(flattened, dtype=np.float32, copy=True))
    validate_observations(result)
    return result, {"kind": "recorded_raw_observation_trace", "path": str(Path(path).resolve()),
        "sha256": sha256(path), "source_sha256": source_hash, "original_shape": shape,
        "samples": len(result), "sampling": "evenly spaced flattened indices, capped at 16384",
        "joint_names": joint_names, "normalization_epsilon": epsilon,
        "metadata": metadata, "physics_evaluation_of_candidate": False,
        "temporal_structure": "unordered calibration rows" if len(shape) == 2 else "time/environment array"}


def verify_changes(original, edited, selected_rows=None):
    selected_rows = TIBIA_ROWS if selected_rows is None else list(selected_rows)
    if len(set(selected_rows)) != len(selected_rows) or any(type(row) is not int or row not in range(18)
                                                          for row in selected_rows):
        raise ValueError("Selected rows must be unique joint indices in [0, 18)")
    if original.keys() != edited.keys():
        raise AssertionError("State keys changed")
    other_rows = [row for row in range(18) if row not in selected_rows]
    for key in original:
        if key in (HEAD_WEIGHT, HEAD_BIAS):
            if not torch.equal(original[key][other_rows], edited[key][other_rows]):
                raise AssertionError("An unselected output row changed")
        elif not same(original[key], edited[key]):
            raise AssertionError(f"Unexpected state change: {key}")
        if isinstance(edited[key], torch.Tensor) and not torch.isfinite(edited[key]).all():
            raise AssertionError(f"Non-finite edited state: {key}")
    changed_rows = [row for row in range(18) if any(not torch.equal(original[key][row], edited[key][row])
                                                 for key in (HEAD_WEIGHT, HEAD_BIAS))]
    return {"only_selected_final_layer_rows_changed": True,
            "only_tibia_final_layer_rows_changed": all(row in TIBIA_ROWS for row in changed_rows),
            "allowed_action_rows": selected_rows, "changed_action_rows": changed_rows,
            "changed_tensor_values": {key: int((original[key] != edited[key]).sum())
                                      for key in (HEAD_WEIGHT, HEAD_BIAS)}}


@torch.inference_mode()
def fit_candidate(state_dict, observations, *, ridge=.01, strength=.2, joint_scope="tibia",
                  protected_cap=.1, max_applied_change=None):
    validate_observations(observations)
    if joint_scope not in ("tibia", "all"):
        raise ValueError("joint_scope must be 'tibia' or explicitly 'all'")
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and positive")
    if not math.isfinite(protected_cap) or not 0 <= protected_cap <= 1:
        raise ValueError("protected_cap must be finite and in [0, 1]")
    applied_cap = strength if max_applied_change is None else max_applied_change
    if not math.isfinite(applied_cap) or not 0 < applied_cap <= 2:
        raise ValueError("max_applied_change must be finite and in (0, 2]")
    actor = FlatActor(state_dict)
    features, raw = actor.features(observations), actor(observations)
    if not torch.isfinite(features).all() or not torch.isfinite(raw).all():
        raise ValueError("Source policy produced non-finite outputs on probes")
    targets = negative_tail_target(raw, strength)
    straight = forward_mask(observations)
    tail_mask = raw < -(1.0 - strength)
    selected_rows = TIBIA_ROWS if joint_scope == "tibia" else list(range(18))
    selected = torch.zeros(18, dtype=torch.bool)
    selected[selected_rows] = True
    if joint_scope == "all":
        non_tibia = torch.ones(18, dtype=torch.bool)
        non_tibia[TIBIA_ROWS] = False
        positive_tail = (raw > 1.0 - strength) & non_tibia[None, :]
        targets = torch.where(positive_tail, -negative_tail_target(-raw, strength), targets)
        tail_mask |= positive_tail
    mask = straight[:, None] & tail_mask & selected[None, :]
    if not mask.any():
        raise ValueError(f"No eligible forward saturation-tail responses for joint_scope={joint_scope}")
    targets = torch.where(mask, targets, raw)
    design = torch.cat((features, torch.ones(len(features), 1)), dim=1).double()
    scale = design.square().mean(dim=0).sqrt().clamp_min(1e-3)
    standardized = design / scale
    coefficients = torch.zeros(18, design.shape[1], dtype=torch.float64)
    for row in selected_rows:
        editable = mask[:, row]
        if not editable.any():
            continue
        protected = ~editable
        weights = editable.double() / editable.sum()
        # Balance protected state classes rather than their number of samples.
        for protection_mask, multiplier in ((protected & straight, 4.), (protected & ~straight, 12.)):
            if protection_mask.any():
                weights += multiplier * protection_mask.double() / protection_mask.sum()
        desired = (targets[:, row] - raw[:, row]).double()
        lhs = standardized.T @ (weights[:, None] * standardized)
        lhs += ridge * torch.eye(design.shape[1], dtype=torch.float64)
        rhs = standardized.T @ (weights * desired)
        coefficients[row] = torch.linalg.solve(lhs, rhs) / scale
    if not torch.isfinite(coefficients).all():
        raise ValueError("Non-finite ridge solution")
    # Bound collateral action changes on the supplied calibration rows. The
    # clipping-aware delta is monotonic in this scalar along each solved row.
    # This is a measured probe constraint, not a global policy bound.
    backoff_factors = {}
    for row in selected_rows:
        delta = design @ coefficients[row]
        old = raw[:, row].double().clamp(-1, 1)
        protected = ~mask[:, row]

        def within_caps(factor):
            applied = (raw[:, row].double() + factor * delta).clamp(-1, 1) - old
            protected_max = applied[protected].abs().max().item() if protected.any() else 0.
            return protected_max <= protected_cap and applied.abs().max().item() <= applied_cap

        factor = 1.
        if not within_caps(factor):
            lower, upper = 0., 1.
            for _ in range(40):
                middle = (lower + upper) / 2.
                if within_caps(middle):
                    lower = middle
                else:
                    upper = middle
            factor = lower * (1. - 1e-5)  # margin for float32 weight serialization
        coefficients[row] *= factor
        backoff_factors[JOINT_NAMES[row]] = factor
    edited = copy.deepcopy(state_dict)
    edited[HEAD_WEIGHT] += coefficients[:, :-1].to(torch.float32)
    edited[HEAD_BIAS] += coefficients[:, -1].to(torch.float32)
    preservation = verify_changes(state_dict, edited, selected_rows)
    candidate = FlatActor(edited)(observations)
    if not torch.isfinite(candidate).all():
        raise ValueError("Candidate produced non-finite outputs")
    applied_delta = candidate.clamp(-1, 1) - raw.clamp(-1, 1)
    clipped_negative = (raw <= -1.) & mask
    clipped_positive = (raw >= 1.) & mask
    released_negative = clipped_negative & (candidate.abs() < 1.)
    released_positive = clipped_positive & (candidate.abs() < 1.)
    released = released_negative | released_positive
    protected_selected = ~mask & selected[None, :]
    before_error = (raw - targets)[mask].square().mean().sqrt().item()
    after_error = (candidate - targets)[mask].square().mean().sqrt().item()
    protected_delta = applied_delta[protected_selected]
    spill = protected_delta.abs().max().item() if protected_delta.numel() else 0.
    legs = {}
    for row in selected_rows:
        leg_mask = mask[:, row]
        legs[JOINT_NAMES[row]] = {"target_samples": int(leg_mask.sum()),
            "negative_clipped_before": int(((raw[:, row] <= -1) & straight).sum()),
            "negative_clipped_after": int(((candidate[:, row] <= -1) & straight).sum()),
            "positive_clipped_before": int(((raw[:, row] >= 1) & straight).sum()),
            "positive_clipped_after": int(((candidate[:, row] >= 1) & straight).sum()),
            "released_originally_clipped": int(released[:, row].sum()),
            "released_negative": int(released_negative[:, row].sum()),
            "released_positive": int(released_positive[:, row].sum()),
            "raw_delta": stats(candidate[:, row] - raw[:, row]),
            "applied_delta": stats(applied_delta[:, row]),
            "protected_applied_delta": stats(applied_delta[~leg_mask, row])}
    criteria = {"target_raw_error_reduced": after_error < before_error,
                "at_least_one_originally_clipped_response_released": bool(released.any()),
                "protected_applied_delta_within_configured_cap": spill <= protected_cap + 1e-6,
                "all_applied_delta_within_configured_cap": applied_delta.abs().max().item() <= applied_cap + 1e-6,
                "unselected_rows_unchanged": bool((applied_delta[:, ~selected] == 0).all())}
    if protected_cap == .1:
        criteria["protected_applied_delta_at_most_010"] = criteria["protected_applied_delta_within_configured_cap"]
    if applied_cap == strength:
        criteria["all_applied_delta_at_most_strength"] = criteria["all_applied_delta_within_configured_cap"]
    if joint_scope == "tibia":
        criteria["non_tibia_rows_unchanged"] = criteria["unselected_rows_unchanged"]
    return edited, {"method": "closed-form float64 feature-scaled final-layer ridge delta",
        "ridge": ridge, "strength": strength, "joint_scope": joint_scope,
        "protected_cap": protected_cap, "max_applied_change": applied_cap,
        "maximum_observed_applied_change": applied_delta.abs().max().item(),
        "uses_default_caps": protected_cap == .1 and applied_cap == strength,
        "cap_scope": "Per-action absolute change on calibration observations only; not a global or physical bound",
        "allowed_action_rows": selected_rows, "changed_action_rows": preservation["changed_action_rows"],
        "target_knee": -(1 - strength),
        "target_negative_asymptote": -(1 - strength / 2),
        "target_positive_asymptote": (1 - strength / 2) if joint_scope == "all" else None,
        "target_applies_to": ("tibia negative tails; coxa/femur both signs; straight forward commands only"
                              if joint_scope == "all" else "tibia negative tails on straight forward commands only"),
        "straight_forward_samples": int(straight.sum()), "target_samples": int(mask.sum()),
        "target_raw_error_before_rms": before_error, "target_raw_error_after_rms": after_error,
        "originally_negative_clipped_targets": int(clipped_negative.sum()),
        "released_originally_negative_clipped_targets": int(released_negative.sum()),
        "originally_positive_clipped_targets": int(clipped_positive.sum()),
        "released_originally_positive_clipped_targets": int(released_positive.sum()),
        "protected_applied_delta": stats(protected_delta), "per_leg": legs,
        "offline_criteria": criteria, "offline_criteria_passed": all(criteria.values()),
        "criteria_scope": "Same input probes used for fitting; neither holdout nor physical validation",
        "calibration_backoff_factors": backoff_factors,
        "preservation": preservation, "coefficient_norm": coefficients.norm().item()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    source_mode = parser.add_mutually_exclusive_group(required=True)
    source_mode.add_argument("--trace", type=Path)
    source_mode.add_argument("--synthetic-probes", action="store_true")
    parser.add_argument("--ridge", type=float, default=.01)
    parser.add_argument("--strength", type=float, default=.2)
    parser.add_argument("--joint-scope", choices=("tibia", "all"), default="tibia",
                        help="Explicit 'all' also permits both signs of coxa/femur tail release")
    parser.add_argument("--protected-cap", type=float, default=.1,
                        help="Maximum protected applied-action change on calibration states, in [0, 1]")
    parser.add_argument("--max-applied-change", type=float, default=None,
                        help="Maximum applied-action change on all calibration states, in (0, 2]; default strength")
    args = parser.parse_args(argv)
    source, destination = args.checkpoint.resolve(), args.output.resolve()
    report_path = destination.with_suffix(".json")
    if source == destination or destination.exists() or report_path.exists():
        raise FileExistsError("Preserve the original and existing outputs; choose a new output path")
    source_hash = sha256(source)
    if source_hash != SOURCE_SHA256:
        raise ValueError("Generator is pinned to the reviewed full-robot model_13250.pt SHA-256")
    torch.set_num_threads(1)
    content = torch.load(source, map_location="cpu", weights_only=True)
    if content.get("iter") != 13250 or "actor_state_dict" not in content:
        raise ValueError("Unexpected checkpoint iteration or actor contract")
    state = content["actor_state_dict"]
    if args.trace:
        observations, evidence = load_trace(args.trace, source_hash)
    else:
        observations, evidence = synthetic_probes(state)
    edited_state, fit = fit_candidate(state, observations, ridge=args.ridge, strength=args.strength,
                                     joint_scope=args.joint_scope, protected_cap=args.protected_cap,
                                     max_applied_change=args.max_applied_change)
    edited = copy.deepcopy(content)
    edited["actor_state_dict"] = edited_state
    for key in content:
        if key != "actor_state_dict" and not same(content[key], edited[key]):
            raise AssertionError(f"Unrelated checkpoint content changed: {key}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        torch.save(edited, stream)
    reloaded = torch.load(destination, map_location="cpu", weights_only=True)
    if not same(edited, reloaded) or sha256(source) != source_hash:
        raise AssertionError("Checkpoint roundtrip or original preservation failed")
    report = {"status": "experimental_saturation_release_candidate_awaiting_physics",
        "source": str(source), "source_sha256": source_hash,
        "output": str(destination), "output_sha256": sha256(destination),
        "generator_sha256": sha256(__file__), "torch_version": str(torch.__version__),
        "contract": {"task": "Hexapod-Walk", "observations": 67, "actions": 18,
            "actor_hidden_dims": [256, 256, 128], "activation": "elu",
            "normalizer": "(raw - mean) / (std + 0.01)", "normalizer_epsilon": EPSILON,
            "normalizer_evidence": "verified exported TorchScript obs_normalizer.eps for this run",
            "joint_names": JOINT_NAMES,
            "action_scales": [.5, 1., 1.] * 6, "clip_actions": 1.0,
            "joint_order_evidence": "full_robot_flat_20261107T162146142Z_1e5c3e24_smoke.json"},
        "changed_actor_tensors": [HEAD_WEIGHT, HEAD_BIAS], "joint_scope": args.joint_scope,
        "allowed_action_rows": fit["allowed_action_rows"], "changed_action_rows": fit["changed_action_rows"],
        "preserved": ["original checkpoint", "normalizer", "upstream actor layers", "unselected output rows",
                      "critic", "exploration distribution", "optimizer", "iteration", "infos"],
        "probe_evidence": evidence, "fit": fit, "serialization_roundtrip_exact": True,
        "original_sha256_unchanged": True, "physics_validated": False, "policy_adopted": False,
        "performance_claim": None,
        "limitations": ["Saturation relief is a hypothesis about joint command utilization, not a normal-gait result.",
            "Stored normalizer moments combine the training history and do not identify gait phases.",
            "Head isolation holds at fixed observations; feedback can change every future action.",
            "No physical candidate rollout or independent holdout has been evaluated.",
            "The retained optimizer is stale for edited weights; this checkpoint is an inference experiment."],
        "required_evaluation": "Paired original/candidate forward, standing and turning rollouts on the same asset, "
            "commands and initial states: per-leg joint range/contact, speed, tracking error, slip and failures."}
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"checkpoint": str(destination), "report": str(report_path),
        "offline_criteria_passed": fit["offline_criteria_passed"], "physics_validated": False}))
    return report


if __name__ == "__main__":
    main()
