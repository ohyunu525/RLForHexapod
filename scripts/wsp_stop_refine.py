"""Trace-conditioned, output-head-only WSP initialization from trained model_999.

Offline IK and damping targets edit weights, not the runtime controller. A saved
candidate is experimental until physical rollout validation. Neither recorded
input fit quality nor reachable IK establishes closed-loop walking or stability.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch

from flat_wsp import FlatActor, HEAD_WEIGHT, HEAD_BIAS, JOINT_NAMES, EPSILON, same, sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "c8828d64d8f1257f9a140a3990cf1250c91c111d8afcb80f1aa4f2f66d648601"
SOURCE = ROOT / "outputs/flat_refine_20261108/trained_model/model_999.pt"
TRACES = ROOT / "outputs/flat_refine_20261108/final_evaluation/candidate"
DEFAULT = np.tile(np.deg2rad([0., 20., -110.]), 6)
SCALE = np.tile([.5, 1., 1.], 6)
HEAD = (HEAD_WEIGHT, HEAD_BIAS)


def rot_rpy(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


class Kinematics:
    """Exact URDF joint origins/+z axes; vectorized body-frame distal-tip FK/J."""
    def __init__(self, path):
        root = ET.parse(path).getroot()
        joints = {j.attrib["name"]: j for j in root.findall("joint")}
        self.fixed, self.origins = [], []
        self.mass = sum(float(link.find("inertial/mass").attrib["value"])
                        for link in root.findall("link") if link.find("inertial/mass") is not None)
        for leg in range(1, 7):
            rotations, positions = [], []
            for part in ("coxa", "femur", "tibia"):
                joint = joints[f"L{leg}_{part}"]
                if not np.allclose(np.fromstring(joint.find("axis").attrib["xyz"], sep=" "), [0, 0, 1]):
                    raise ValueError("Only reviewed URDF +z revolute axes supported")
                origin = joint.find("origin")
                rotations.append(rot_rpy(np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")))
                positions.append(np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" "))
            self.fixed.append(rotations); self.origins.append(positions)
        self.fixed, self.origins = np.asarray(self.fixed), np.asarray(self.origins)

    def fk_jacobian(self, leg, q):
        q = np.atleast_2d(q).astype(float)
        n = len(q); rotation = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        position = np.zeros((n, 3)); axes, centers = [], []
        for joint in range(3):
            position += np.einsum("nij,j->ni", rotation, self.origins[leg, joint])
            rotation = rotation @ self.fixed[leg, joint]
            centers.append(position.copy()); axes.append(rotation[:, :, 2].copy())
            c, s = np.cos(q[:, joint]), np.sin(q[:, joint])
            rz = np.zeros((n, 3, 3)); rz[:, 0, 0] = rz[:, 1, 1] = c
            rz[:, 1, 0] = s; rz[:, 0, 1] = -s; rz[:, 2, 2] = 1.
            rotation = rotation @ rz
        offset = [-.004, -.147, -.021] if leg == 0 else [.004, .147, -.021]
        tip = position + np.einsum("nij,j->ni", rotation, offset)
        jac = np.stack([np.cross(axis, tip-center) for axis, center in zip(axes, centers)], axis=-1)
        return tip, jac

    def anchor(self, height):
        # Match measured support-tip surface height (~12 mm), with a wide low
        # stance and narrower high stance to keep femur actions below clipping.
        if not .17-1e-8 <= height <= .24+1e-8:
            raise ValueError("Anchor height outside reviewed [.17,.24] command range")
        reach = .165 + 1.8*(.205-height)
        result, tips, torques = [], [], []
        for leg in range(6):
            outward = self.origins[leg, 0, :2] / np.linalg.norm(self.origins[leg, 0, :2])
            q = np.deg2rad([0., 65., -140.])
            for _ in range(60):
                tip, jac = self.fk_jacobian(leg, q); tip, jac = tip[0], jac[0]
                error = np.array([(tip[:2]-self.origins[leg, 0, :2])@outward-reach, tip[2]-(.012-height)])
                if np.linalg.norm(error) < 1e-10:
                    break
                j = np.stack((outward @ jac[:2, 1:], jac[2, 1:]))
                q[1:] -= np.clip(np.linalg.solve(j, error), -.15, .15)
            if np.linalg.norm(error) > 1e-7:
                raise ValueError("Standing IK failed to converge")
            result.append(q); tips.append(tip)
            torques.append(jac[2]*self.mass*9.81/6.)
        q = np.asarray(result).reshape(18)
        action = (q-DEFAULT)/SCALE
        if np.abs(action).max() >= .98:
            raise ValueError("Standing anchor lacks action margin")
        return q, {"height_m": float(height), "outward_reach_m": reach,
                   "joint_positions_rad": q.tolist(), "action": action.tolist(),
                   "tip_positions_body_m": np.asarray(tips).tolist(),
                   "foot_load_only_joint_torque_estimate_Nm": np.asarray(torques).tolist(),
                   "torque_estimate_scope": "J^T times equal mg/6 upward foot loads; excludes link gravity, dynamics and unequal loads; not a stability/actuator-margin proof."}


def mirror_action(a):
    out = np.asarray(a).reshape(-1, 6, 3)[:, ::-1].copy()
    out[:, :, 0] *= -1
    return out.reshape(-1, 18)


def mirror_obs(obs):
    out = obs.copy()
    for start in (0, 18, 49):
        out[:, start:start+18] = mirror_action(obs[:, start:start+18])
    out[:, [37, 39, 41, 43, 46, 47]] *= -1
    return out


def body_up(quat):
    x, y, z, w = np.moveaxis(quat, -1, 0)
    return np.stack((2*(x*z+y*w), 2*(y*z-x*w), 1-2*(x*x+y*y)), -1)


def world_up_in_body(quat):
    x, y, z, w = np.moveaxis(quat, -1, 0)
    return np.stack((2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1)


def damping_delta(kin, obs, pre_quat, support, yaw_damping, vertical_damping, tip_cap=.002):
    """Small offline stance-foot displacement labels, not dynamic identification.

    Positive yaw error -> stance feet move along +z cross r in the body frame,
    whose rigid-contact kinematic body reaction opposes excess positive yaw.
    Positive world vz -> feet retract upward to oppose upward body motion.
    The full URDF Jacobian maps those displacements to joint position changes.
    """
    q = obs[:, :18] + DEFAULT
    up_b = world_up_in_body(pre_quat)
    vz_w = (obs[:, 36:39]*up_b).sum(-1)
    yaw_error = obs[:, 41]-obs[:, 47]
    delta = np.zeros_like(q)
    for leg in range(6):
        sl = slice(3*leg, 3*leg+3)
        tip, jac = kin.fk_jacobian(leg, q[:, sl])
        yaw_axis_cross_tip = np.stack((-tip[:, 1], tip[:, 0], np.zeros(len(tip))), -1)
        dp = yaw_damping*yaw_error[:, None]*yaw_axis_cross_tip + vertical_damping*vz_w[:, None]*up_b
        dp *= np.minimum(1., tip_cap/np.maximum(np.linalg.norm(dp, axis=1), 1e-12))[:, None]
        dp *= support[:, leg:leg+1]
        # Damped inverse avoids large actions near singular leg configurations.
        dq = np.einsum("nji,nj->ni", jac, np.linalg.solve(jac@np.swapaxes(jac, 1, 2)+1e-5*np.eye(3), dp))
        # Uniform scaling preserves the Cartesian damping direction; clipping
        # individual joints can break cancellation and enlarge tip displacement.
        dq *= np.minimum(1., .025/np.maximum(np.abs(dq).max(axis=1), 1e-12))[:, None]
        delta[:, sl] = dq/SCALE[sl]
    return delta


def stop_targets(obs, anchor_map, velocity_damping):
    if len(obs) == 0:
        return np.empty((0, 18), dtype=float)
    q = np.stack([anchor_map[round(float(h), 6)] for h in obs[:, 48]])
    # Position anchoring plus a small offline braking label. Motor PD remains
    # the original one; there is no runtime override or command gate added.
    target = (q-DEFAULT-velocity_damping*obs[:, 18:36])/SCALE
    return np.clip(target, -.98, .98)


@torch.inference_mode()
def outputs(actor, obs):
    return actor(torch.as_tensor(obs, dtype=torch.float32)).numpy().astype(float)


def load_recordings(folder, actor, kin, args, anchor_map):
    blocks, heldout, provenance = [], [], []
    counterfactual = []
    for path in sorted(folder.glob("*.npz")):
        report = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if (report.get("status") != "completed" or report.get("trace_sha256") != sha256(path)
                or report.get("checkpoint_sha256") != SOURCE_SHA):
            raise ValueError(f"Incomplete/corrupt calibration trace: {path}")
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        if any(str(data[key].item()) != SOURCE_SHA for key in ("checkpoint_sha256", "source_sha256")):
            raise ValueError("Trace policy differs from source999")
        if data["joint_names"].tolist() != JOINT_NAMES or not np.isclose(float(data["normalization_epsilon"]), EPSILON):
            raise ValueError("Trace joint order/normalizer mismatch")
        if data["terminated"].any() or data["obs"].shape[2:] != (67,):
            raise ValueError("Calibration requires complete finite nonfailed 67-input trajectories")
        obs = data["obs"].reshape(-1, 67).astype(float)
        if not np.isfinite(obs).all():
            raise ValueError("Non-finite calibration observations")
        raw = outputs(actor, obs)
        replay = float(np.max(np.abs(raw-data["actions_raw"].reshape(-1, 18))))
        if replay > 2e-4:
            raise ValueError(f"Source action replay mismatch {replay}")
        t, n = data["episode_id"].shape
        moving = (np.linalg.norm(obs[:, 45:47], axis=1) > .025) | (np.abs(obs[:, 47]) > .1)
        frame = np.repeat(np.arange(t), n)
        train = frame < int(.75*t)
        # Trace fields are post-action; shift contacts back one frame to align
        # with policy observations. Reset boundaries/first frames have no label.
        tip, force = data["foot_tip_pos_w"], data["tibia_force_w"]
        post_support = (force[..., 2]>1.) & (tip[..., 2]>=-.01) & (tip[..., 2]<=.03)
        support = np.zeros_like(post_support)
        support[1:] = post_support[:-1] & (data["episode_id"][1:]==data["episode_id"][:-1])[..., None]
        target = raw.copy()
        target[moving] += damping_delta(kin, obs[moving], data["pre_root_pose"].reshape(-1, 7)[moving, 3:7],
                                        support.reshape(-1, 6)[moving], args.yaw_damping, args.vertical_damping)
        for h in np.unique(np.round(obs[:, 48], 6)):
            anchor_map[float(h)] = kin.anchor(float(h))[0]
        stand = ~moving
        desired_stop = stop_targets(obs[stand], anchor_map, args.stand_velocity_damping)
        target[stand] += args.stop_strength*(desired_stop-raw[stand])
        for label, select, weight in (("move", moving, args.moving_weight), ("stop", stand, args.stop_weight)):
            for destination, split in ((blocks, train), (heldout, ~train)):
                chosen = select & split
                if chosen.any():
                    destination.append({"name": path.stem+"/"+label, "obs": obs[chosen], "target": target[chosen],
                                        "raw": raw[chosen], "moving": label=="move", "weight": weight})
        # Physically recorded poses/velocities under a counterfactual zero
        # command cover walk-to-stop across heights; these are not new rollouts.
        cf = obs[moving & train][::8].copy(); cf[:, 45:48] = 0.
        if len(cf):
            counterfactual.append(cf)
        provenance.append({"path": str(path.resolve()), "sha256": sha256(path), "samples": len(obs),
                           "train_frames": [0, int(.75*t)], "heldout_frames": [int(.75*t), t],
                           "source_action_replay_max_abs_error": replay})
    required = {"forward10", "forward15", "forward20", "forward_low", "forward_high", "stand", "turn_left", "turn_right", "walk_stop"}
    if {Path(p["path"]).stem for p in provenance} != required:
        raise ValueError("Require the complete reviewed nine-case calibration set")
    cf = np.concatenate(counterfactual)
    raw = outputs(actor, cf)
    target = raw+args.stop_strength*(stop_targets(cf, anchor_map, args.stand_velocity_damping)-raw)
    blocks.append({"name": "counterfactual_stop", "obs": cf, "target": target, "raw": raw,
                   "moving": False, "weight": args.stop_weight})
    return blocks, heldout, provenance


def synthetic_anchors(actor, kin, args, anchor_map):
    rng = np.random.default_rng(args.seed)
    blocks = []
    for height in np.linspace(.17, .24, 9):
        key = round(float(height), 6); q, _ = kin.anchor(height); anchor_map[key] = q
        obs = np.zeros((32, 67)); obs[:, :18] = q-DEFAULT
        obs[1:, :18] += rng.normal(0., .02, (31, 18))
        obs[1:, 18:36] = rng.normal(0., .12, (31, 18))
        obs[:, 44] = -1.; obs[:, 48] = key; obs[:, 49:67] = (q-DEFAULT)/SCALE
        raw = outputs(actor, obs)
        target = raw+args.stop_strength*(stop_targets(obs, anchor_map, args.stand_velocity_damping)-raw)
        blocks.append({"name": f"synthetic_equilibrium_{key}", "obs": obs, "target": target, "raw": raw,
                       "moving": False, "weight": args.stop_weight/9})
        # Protect starting/stopping command sensitivity near the new rest pose.
        start = np.repeat(obs[:8], 8, axis=0)
        start[:, 45:48] = np.tile([[.03,0,0],[.1,0,0],[.2,0,0],[-.1,0,0],
                                  [0,.05,0],[0,-.05,0],[0,0,.25],[0,0,-.25]], (8,1))
        out = outputs(actor, start)
        blocks.append({"name": f"synthetic_moving_{key}", "obs": start, "target": out, "raw": out,
                       "moving": True, "weight": args.moving_weight/9})
    return blocks


@torch.inference_mode()
def fit_head(state, blocks, ridge, moving_cap):
    actor = FlatActor(state)
    obs = np.concatenate([b["obs"] for b in blocks])
    raw = np.concatenate([b["raw"] for b in blocks]); target = np.concatenate([b["target"] for b in blocks])
    weights = np.concatenate([np.full(len(b["obs"]), b["weight"]/len(b["obs"])) for b in blocks])
    design = torch.cat([actor.features(torch.as_tensor(obs, dtype=torch.float32)), torch.ones(len(obs), 1)], 1).double()
    scale = design.square().mean(0).sqrt().clamp_min(.01)
    x = design/scale; weight = torch.as_tensor(weights/weights.sum(), dtype=torch.float64)
    lhs = x.T@(weight[:, None]*x)+ridge*torch.eye(129, dtype=torch.float64)
    rhs = x.T@(weight[:, None]*torch.as_tensor(target-raw, dtype=torch.float64))
    delta = (torch.linalg.solve(lhs, rhs)/scale[:, None]).T
    protective_obs = np.concatenate([b["obs"] for b in blocks if b["moving"]])
    protective_raw = outputs(actor, protective_obs)
    h = torch.cat([actor.features(torch.as_tensor(protective_obs, dtype=torch.float32)), torch.ones(len(protective_obs), 1)], 1).double()
    change = (h@delta.T).numpy()
    factors = []
    for joint in range(18):
        old, d = protective_raw[:, joint], change[:, joint]
        def inside(factor):
            return (np.max(np.abs(factor*d)) <= 1.6*moving_cap
                    and np.max(np.abs(np.clip(old+factor*d, -1, 1)-np.clip(old, -1, 1))) <= .8*moving_cap)
        lo, hi = 0., 1.
        if inside(1.):
            lo = 1.
        else:
            for _ in range(36):
                mid = (lo+hi)/2
                if inside(mid): lo = mid
                else: hi = mid
        delta[joint] *= lo; factors.append(lo)
    # Enforce a per-case RMS guard too, not only a pooled/global maximum.
    global_factor = 1.
    for block in blocks:
        if not block["moving"]: continue
        bh = torch.cat([actor.features(torch.as_tensor(block["obs"], dtype=torch.float32)), torch.ones(len(block["obs"]), 1)], 1).double()
        d = (bh@delta.T).numpy()
        rms = float(np.sqrt(np.mean(d*d)))
        if rms > 0: global_factor = min(global_factor, (.8*moving_cap/3)/rms)
    delta *= global_factor
    edited = copy.deepcopy(state)
    edited[HEAD_WEIGHT] += delta[:, :128].float(); edited[HEAD_BIAS] += delta[:, 128].float()
    return edited, {"ridge": ridge, "feature_condition_number": float(torch.linalg.cond(lhs)),
                   "per_row_backoff": factors, "global_rms_backoff": global_factor,
                   "changed_head_delta_l2": float(delta.norm()), "training_rows": len(obs),
                   "heldout_used_for_fit_or_backoff": False}


def evaluate_blocks(source, edited, blocks, moving_cap):
    model, rows, valid = FlatActor(edited), [], True
    for block in blocks:
        out = outputs(model, block["obs"]); old = block["raw"]
        raw_delta = out-old; applied_delta = np.clip(out, -1, 1)-np.clip(old, -1, 1)
        row = {"name": block["name"], "moving": block["moving"], "samples": len(out),
               "raw_delta_rms": float(np.sqrt(np.mean(raw_delta**2))), "raw_delta_max": float(np.abs(raw_delta).max()),
               "applied_delta_rms": float(np.sqrt(np.mean(applied_delta**2))), "applied_delta_max": float(np.abs(applied_delta).max()),
               "source_target_rmse": float(np.sqrt(np.mean((old-block["target"])**2))),
               "candidate_target_rmse": float(np.sqrt(np.mean((out-block["target"])**2))),
               "raw_clip_fraction": float((np.abs(out)>=1).mean()), "finite": bool(np.isfinite(out).all())}
        row["moving_protection_passed"] = (not block["moving"] or (row["raw_delta_max"] <= 2*moving_cap+1e-6
            and row["applied_delta_max"] <= moving_cap+1e-6 and row["raw_delta_rms"] <= moving_cap/3+1e-6))
        valid &= row["finite"] and row["moving_protection_passed"]
        rows.append(row)
    return rows, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=SOURCE)
    parser.add_argument("--traces", type=Path, default=TRACES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-strength", type=float, default=1.)
    parser.add_argument("--yaw-damping", type=float, default=0.)
    parser.add_argument("--vertical-damping", type=float, default=0.)
    parser.add_argument("--stand-velocity-damping", type=float, default=.01)
    parser.add_argument("--moving-cap", type=float, default=.08)
    parser.add_argument("--ridge", type=float, default=.001)
    parser.add_argument("--stop-weight", type=float, default=4.)
    parser.add_argument("--moving-weight", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=999)
    args = parser.parse_args()
    if not 0 < args.stop_strength <= 1 or not 0 < args.moving_cap <= .4 or args.ridge <= 0:
        parser.error("Require stop-strength in(0,1], moving-cap in(0,.4], positive ridge")
    for key in ("yaw_damping", "vertical_damping", "stand_velocity_damping", "stop_weight", "moving_weight"):
        if not np.isfinite(getattr(args, key)) or getattr(args, key)<0:
            parser.error(f"Invalid {key}")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("Refusing to overwrite WSP artifacts")
    torch.set_num_threads(1)
    if sha256(args.checkpoint) != SOURCE_SHA:
        raise ValueError("Pinned source model_999 hash mismatch")
    content = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = content["actor_state_dict"]; actor = FlatActor(state)
    if content["iter"] != 999:
        raise ValueError("Unexpected source iteration")
    urdf = ROOT/"src/RLforHexapod/assets/urdf/robot.urdf"; kin = Kinematics(urdf)
    anchor_map = {}; blocks, heldout, traces = load_recordings(args.traces, actor, kin, args, anchor_map)
    blocks.extend(synthetic_anchors(actor, kin, args, anchor_map))
    edited_state, fit = fit_head(state, blocks, args.ridge, args.moving_cap)
    training, train_ok = evaluate_blocks(state, edited_state, blocks, args.moving_cap)
    validation, heldout_ok = evaluate_blocks(state, edited_state, heldout, args.moving_cap)
    if not train_ok or not heldout_ok:
        raise ValueError("Frozen-input moving protection failed; no checkpoint written: "+json.dumps(validation))
    edited = copy.deepcopy(content); edited["actor_state_dict"] = edited_state
    changed = [key for key in state if not same(state[key], edited_state[key])]
    if changed != list(HEAD) or any(not same(content[k], edited[k]) for k in content if k!="actor_state_dict"):
        raise AssertionError("Unexpected parameter/metadata changes")
    if any(not same(state[k], edited_state[k]) for k in state if k not in HEAD):
        raise AssertionError("Upstream/distribution/normalizer changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream: torch.save(edited, stream)
    if not same(edited, torch.load(args.output, map_location="cpu", weights_only=True)):
        raise AssertionError("Checkpoint serialization mismatch")
    if sha256(args.checkpoint) != SOURCE_SHA: raise AssertionError("Source changed")
    anchors = [kin.anchor(h)[1] for h in (.17, .205, .24)]
    report = {"status": "experimental_weight_surgery_awaiting_physics", "source_sha256": SOURCE_SHA,
        "source_checkpoint": str(args.checkpoint.resolve()), "candidate": str(args.output.resolve()),
        "candidate_sha256": sha256(args.output), "source_sha256_unchanged": True,
        "normalization_epsilon": EPSILON, "normalizer_preserved": True, "upstream_preserved": True,
        "changed_actor_tensors": changed, "runtime_inputs_added": False, "runtime_controller_added": False,
        "all_other_checkpoint_fields_preserved": True, "roundtrip_exact": True, "physics_validated": False,
        "generator_sha256": sha256(__file__), "actor_helper_sha256": sha256(ROOT/"scripts/flat_wsp.py"),
        "urdf_sha256": sha256(urdf), "recorded_traces": traces,
        "settings": {k:v for k,v in vars(args).items() if not isinstance(v, Path)},
        "fit": fit, "training": training, "heldout": validation,
        "anchor_contract": {"heights": [.17,.205,.24], "joint_names": JOINT_NAMES,
            "joint_positions_rad": [a["joint_positions_rad"] for a in anchors],
            "source_sha256": SOURCE_SHA, "physical_checked": False, "anchors": anchors},
        "target_scope": "Static IK and optional stance-only yaw/vz damping are offline labels. Moving source gait is protected on recorded/synthetic inputs; no motion-phase or command gate is added at runtime.",
        "limitations": ["A small output change on frozen observations does not guarantee speed or stability after feedback changes state.",
            "A reachable standing target does not establish stable standing or eliminate marching in closed loop.",
            "Damping directions follow rigid-contact kinematics; they are not causally identified dynamic effects.",
            "Stop fit may remain weak when preservation caps limit the shared output-head representation; inspect reported target errors.",
            "Synthetic/counterfactual states are labeled separately from observed physics; heldout rows are later parts of the same episodes.",
            "Preserved optimizer/value states are stale for edited weights; training must actor-only warm-start with fresh optimizer/value model."]}
    with args.output.with_suffix(".json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"candidate": str(args.output), "candidate_sha256": report["candidate_sha256"],
                      "fit": fit, "heldout": validation, "physics_validated": False}), flush=True)


if __name__ == "__main__":
    main()
