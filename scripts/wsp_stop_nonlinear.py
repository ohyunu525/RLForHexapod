"""Offline nonlinear WSP target fitting in the existing actor's final two layers.

The source normalization and first two hidden layers are frozen exactly. No
runtime phase, command gate, controller, architecture or observation is added.
This is supervised weight-space initialization, not PPO or physical validation.
The last 25% of every recorded trajectory is held out and never fitted or used
to choose an optimization checkpoint. Moving-input guards select using training
rows only; heldout protection is checked once before writing a candidate.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

import wsp_stop_refine as base


EDIT_KEYS = ["mlp.4.weight", "mlp.4.bias", "mlp.6.weight", "mlp.6.bias"]


class EditableTail(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        self.hidden = torch.nn.Linear(256, 128)
        self.head = torch.nn.Linear(128, 18)
        with torch.no_grad():
            for module, layer in ((self.hidden, 4), (self.head, 6)):
                module.weight.copy_(state[f"mlp.{layer}.weight"])
                module.bias.copy_(state[f"mlp.{layer}.bias"])

    def forward(self, prefix):
        return self.head(torch.nn.functional.elu(self.hidden(prefix)))

    def export(self, source):
        result = copy.deepcopy(source)
        for module, layer in ((self.hidden, 4), (self.head, 6)):
            result[f"mlp.{layer}.weight"] = module.weight.detach().clone()
            result[f"mlp.{layer}.bias"] = module.bias.detach().clone()
        return result


@torch.inference_mode()
def prefix_features(actor, raw):
    obs = torch.as_tensor(raw, dtype=torch.float32)
    return actor.mlp[:4]((obs-actor.mean)/(actor.std+base.EPSILON)).detach()


def prepare(state, blocks):
    actor = base.FlatActor(state)
    # Clone outside inference_mode so autograd may save these constant inputs
    # for trainable tail weights, including if batching is changed to a view.
    prefix = prefix_features(actor, np.concatenate([b["obs"] for b in blocks])).clone()
    raw = torch.tensor(np.concatenate([b["raw"] for b in blocks]), dtype=torch.float32)
    target = torch.tensor(np.concatenate([b["target"] for b in blocks]), dtype=torch.float32)
    moving = torch.tensor(np.concatenate([np.full(len(b["obs"]), b["moving"]) for b in blocks]), dtype=torch.bool)
    weights = torch.tensor(np.concatenate([np.full(len(b["obs"]), b["weight"]/len(b["obs"])) for b in blocks]), dtype=torch.float32)
    groups, start = [], 0
    for b in blocks:
        groups.append((b["name"], b["moving"], start, start+len(b["obs"])))
        start += len(b["obs"])
    return {"prefix": prefix, "raw": raw, "target": target, "moving": moving,
            "weights": weights, "groups": groups}


@torch.inference_mode()
def training_score(model, dataset, cap):
    out = model(dataset["prefix"])
    raw_delta = out-dataset["raw"]
    applied_delta = out.clamp(-1,1)-dataset["raw"].clamp(-1,1)
    valid, max_raw, max_applied, max_case_rms = bool(torch.isfinite(out).all()), 0., 0., 0.
    for _, moving, start, end in dataset["groups"]:
        if not moving: continue
        rd, ad = raw_delta[start:end], applied_delta[start:end]
        rmax, amax, rms = float(rd.abs().max()), float(ad.abs().max()), float(rd.square().mean().sqrt())
        max_raw, max_applied, max_case_rms = max(max_raw,rmax), max(max_applied,amax), max(max_case_rms,rms)
        valid &= rmax <= 1.6*cap and amax <= .8*cap and rms <= .8*cap/3
    stop = ~dataset["moving"]
    weight = dataset["weights"][stop]
    error = (out[stop]-dataset["target"][stop]).square().mean(1)
    loss = float((weight*error).sum()/weight.sum())
    moving=dataset["moving"];move_weight=dataset["weights"][moving]
    move_error=((out[moving]-dataset["target"][moving]).square().mean(1)
                +(out[moving].clamp(-1,1)-dataset["target"][moving].clamp(-1,1)).square().mean(1))
    move_loss=float((move_weight*move_error).sum()/move_weight.sum())
    return {"training_guard_passed": valid, "weighted_stop_target_rmse": loss**.5,
            "weighted_moving_target_loss":move_loss,
            "maximum_moving_raw_delta": max_raw, "maximum_moving_applied_delta": max_applied,
            "maximum_case_moving_delta_rms": max_case_rms}


def selection_value(score, initial, args, candidate=True):
    """Training-only selection; joint repair may trade a bounded stop-fit change."""
    if not score["training_guard_passed"]:return float("inf")
    if args.selection_objective=="stop":return score["weighted_stop_target_rmse"]
    if score["weighted_stop_target_rmse"]>args.stop_rmse_max_factor*initial["weighted_stop_target_rmse"]:
        return float("inf")
    if candidate and score["weighted_moving_target_loss"]>=initial["weighted_moving_target_loss"]:
        return float("inf")
    return score["weighted_stop_target_rmse"]**2+args.moving_weight*score["weighted_moving_target_loss"]


def moving_objective(output, source, target, cap):
    """Fit optional small damping labels while guarding deviation from SOURCE."""
    fit_loss = ((output-target).square().mean()
                +(output.clamp(-1,1)-target.clamp(-1,1)).square().mean())
    raw_deviation = output-source
    applied_deviation = output.clamp(-1,1)-source.clamp(-1,1)
    guard_loss = ((raw_deviation.abs()-1.6*cap).clamp_min(0).square().mean()
                  +(applied_deviation.abs()-.8*cap).clamp_min(0).square().mean())
    return fit_loss, guard_loss


def read_extra_trace(path, producer_hash, reference_report, moving):
    """Validate actual producer identity and command scope before assigning labels."""
    report=json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if (report.get("status")!="completed" or report.get("checkpoint_sha256")!=producer_hash
            or report.get("trace_sha256")!=base.sha256(path)):
        raise ValueError("Extra trace report/producer/trace identity mismatch")
    with np.load(path,allow_pickle=False) as archive:data={k:archive[k] for k in archive.files}
    if any(str(data[k].item())!=producer_hash for k in ("checkpoint_sha256","source_sha256")):
        raise ValueError("Extra trace producer differs from explicit hash")
    if data["joint_names"].tolist()!=base.JOINT_NAMES or not np.isclose(float(data["normalization_epsilon"]),base.EPSILON):
        raise ValueError("Extra trace joint/normalizer contract mismatch")
    obs=data["obs"]
    if (obs.ndim!=3 or obs.shape[2]!=67 or not np.isfinite(obs).all()
            or data["terminated"].shape!=obs.shape[:2] or data["terminated"].any()):
        raise ValueError("Extra trace requires finite, nonfailed recorded 67-input episodes")
    if len(obs)!=report["steps_requested"] or report.get("num_envs",obs.shape[1])!=obs.shape[1]:
        raise ValueError("Extra trace must be complete with matching environment count")
    command=obs[...,45:48]
    report_command=np.array([report["command"][k] for k in ("vx","vy","yaw_rate")])
    if not np.allclose(command[0],report_command,atol=1e-7,rtol=0):
        raise ValueError("Extra trace initial command differs from report")
    if moving:
        is_moving=(np.linalg.norm(command[...,:2],axis=-1)>.025)|(np.abs(command[...,2])>.1)
        if not is_moving.all():raise ValueError("Extra moving trace contains stationary command rows")
    elif not np.allclose(command,0.,atol=1e-7,rtol=0):
        raise ValueError("Extra stop trace must contain only zero velocity commands")
    for key in ("command_actual","command_target"):
        if key in data and (data[key].shape!=obs.shape[:2]+(3,) or not np.allclose(data[key],command,atol=1e-7,rtol=0)):
            raise ValueError("Extra trace live/planned commands disagree with observations")
    for key in ("asset_usd_sha256","asset_manifest_sha256","environment_source_sha256","step_dt_s"):
        if key not in report or report[key]!=reference_report[key]:raise ValueError(f"Extra trace physics contract differs: {key}")
    if int(.75*len(obs))==0 or int(.75*len(obs))==len(obs):
        raise ValueError("Extra trace needs samples in both temporal partitions")
    return obs


def load_extra_stop(path, producer_hash, actor, kin, anchor_map, weight, reference_report):
    """Use new recorded policy states without pretending their actions are source999's."""
    obs=read_extra_trace(path,producer_hash,reference_report,moving=False)
    flat=obs.reshape(-1,67).astype(float)
    for height in np.unique(np.round(flat[:,48],6)):anchor_map[float(height)]=kin.anchor(float(height))[0]
    raw=base.outputs(actor,flat)
    target=base.stop_targets(flat,anchor_map,.01)
    split=int(.75*len(obs));cut=split*obs.shape[1]
    if cut==0 or cut==len(flat):raise ValueError("Extra stop needs samples in both temporal partitions")
    def block(start,end,label):
        return {"name":"on_policy_stop/"+path.parent.name+"/"+label,"obs":flat[start:end],
                "raw":raw[start:end],"target":target[start:end],"moving":False,"weight":weight}
    provenance={"path":str(path.resolve()),"sha256":base.sha256(path),"recording_policy_sha256":producer_hash,
        "label_base_policy_sha256":base.SOURCE_SHA,"observed_trajectories":True,
        "train_frames":[0,split],"heldout_frames":[split,len(obs)],"samples":len(flat),
        "source_action_replay_asserted":False,
        "scope":"Recorded extra-policy states; source999 actions are recomputed only as an output-deviation reference, not claimed to be the recorded producer actions."}
    return block(0,cut,"train"),block(cut,len(flat),"heldout"),provenance


def load_extra_moving(path, producer_hash, actor, weight, reference_report):
    """Distill source999 outputs on states visited by a different moving actor."""
    obs=read_extra_trace(path,producer_hash,reference_report,moving=True)
    flat=obs.reshape(-1,67).astype(float);raw=base.outputs(actor,flat)
    split=int(.75*len(obs));cut=split*obs.shape[1]
    def block(start,end,label):
        return {"name":"on_policy_move/"+path.parent.name+"/"+path.stem+"/"+label,
                "obs":flat[start:end],"raw":raw[start:end],"target":raw[start:end],"moving":True,"weight":weight}
    provenance={"path":str(path.resolve()),"sha256":base.sha256(path),"recording_policy_sha256":producer_hash,
        "label_base_policy_sha256":base.SOURCE_SHA,"observed_trajectories":True,
        "train_frames":[0,split],"heldout_frames":[split,len(obs)],"samples":len(flat),
        "source_action_replay_asserted":False,"target":"Original source999 actor mean on the recorded producer's states",
        "scope":"The recording producer visited these moving states; source999 provides recomputed distillation labels, not recorded actions. All ordinary moving guards remain against source999."}
    return block(0,cut,"train"),block(cut,len(flat),"heldout"),provenance


def validated_initial_tail(path, source):
    if path is None:return source,{"kind":"source999"}
    report=json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if report.get("source_sha256")!=base.SOURCE_SHA or report.get("candidate_sha256")!=base.sha256(path):
        raise ValueError("Tail warm-start lineage hash mismatch")
    if report.get("edit_scope")!="last_two_layers" or not report.get("prefix_preserved"):
        raise ValueError("Tail warm-start must have a verified final-two-layer scope")
    state=torch.load(path,map_location="cpu",weights_only=True)["actor_state_dict"]
    if source.keys()!=state.keys():raise ValueError("Tail warm-start actor keys differ")
    changed=[key for key in source if not base.same(source[key],state[key])]
    if set(changed)!=set(report.get("changed_actor_tensors",[])) or not set(changed)<=set(EDIT_KEYS):
        raise ValueError("Tail warm-start changed unexpected tensors")
    for key in source:
        if source[key].shape!=state[key].shape or source[key].dtype!=state[key].dtype or not torch.isfinite(state[key]).all():
            raise ValueError("Tail warm-start tensor contract differs")
    return state,{"kind":"verified_final_two_layer_candidate","path":str(path.resolve()),"sha256":base.sha256(path)}


def synthetic_stop_neighborhoods(actor, kin, label_args, anchor_map, jitter, weight, samples):
    """Broaden only zero-command rows; keep legacy moving protection bitwise.

    These independent perturbations are synthetic coverage, not simulated
    trajectories or a proof that the fitted map is contractive in physics.
    """
    blocks=base.synthetic_anchors(actor,kin,label_args,anchor_map)
    rng=np.random.default_rng(label_args.seed+171)
    for block in blocks:
        if block["moving"]:continue
        block["weight"]*=weight
        if jitter is None:continue
        q_std, qdot_std, previous_std, omega_std, velocity_std=jitter
        height=float(block["obs"][0,48]);q=anchor_map[round(height,6)]
        obs=np.zeros((samples,67));obs[:,:18]=q-base.DEFAULT
        obs[:,44]=-1.;obs[:,48]=height;obs[:,49:67]=(q-base.DEFAULT)/base.SCALE
        # Keep an exact equilibrium as row zero. The remaining samples vary
        # every feedback channel independently so labels do not encode a gait.
        obs[1:,:18]+=rng.normal(0.,q_std,(samples-1,18))
        obs[1:,18:36]=rng.normal(0.,qdot_std,(samples-1,18))
        obs[1:,36:39]=rng.normal(0.,velocity_std,(samples-1,3))
        obs[1:,39:42]=rng.normal(0.,omega_std,(samples-1,3))
        obs[1:,49:67]+=rng.normal(0.,previous_std,(samples-1,18))
        obs[:,49:67]=np.clip(obs[:,49:67],-1.,1.)
        raw=base.outputs(actor,obs)
        target=raw+label_args.stop_strength*(base.stop_targets(obs,anchor_map,label_args.stand_velocity_damping)-raw)
        block.update(obs=obs,raw=raw,target=target)
    return blocks


def optimize(state, blocks, args, initial_state=None):
    dataset = prepare(state, blocks)
    model = EditableTail(state if initial_state is None else initial_state)
    original = [p.detach().clone() for p in model.parameters()]
    with torch.inference_mode():
        replay = float((EditableTail(state)(dataset["prefix"])-dataset["raw"]).abs().max())
    if replay > 2e-4:
        raise ValueError(f"Frozen-prefix split does not reproduce source output: {replay}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(args.seed)
    moving_ids = torch.nonzero(dataset["moving"], as_tuple=True)[0]
    stop_ids = torch.nonzero(~dataset["moving"], as_tuple=True)[0]
    moving_probability = dataset["weights"][moving_ids]
    stop_probability = dataset["weights"][stop_ids]
    initial = training_score(model, dataset, args.moving_cap)
    print(json.dumps({"offline_step":0,"selection_objective":args.selection_objective,
                      "joint_selection_stop_rmse_limit":args.stop_rmse_max_factor*initial["weighted_stop_target_rmse"],
                      **initial}),flush=True)
    best_score, best_step, best_state = selection_value(initial,initial,args,candidate=False), 0, copy.deepcopy(model.state_dict())
    initial_tail=copy.deepcopy(model.state_dict());best_fraction=1.
    history = [{"step": 0, **initial}]
    started = time.monotonic()
    for step in range(1, args.steps+1):
        mid = moving_ids[torch.multinomial(moving_probability, args.batch_size//2, replacement=True, generator=generator)]
        sid = stop_ids[torch.multinomial(stop_probability, args.batch_size//2, replacement=True, generator=generator)]
        output_move = model(dataset["prefix"][mid]); output_stop = model(dataset["prefix"][sid])
        # Preserve the source moving mean, including its unclipped saturation
        # margins; near-zero command behavior is learned inside existing ELUs.
        move_loss, cap_loss = moving_objective(output_move,dataset["raw"][mid],dataset["target"][mid],args.moving_cap)
        stop_loss = (output_stop-dataset["target"][sid]).square().mean()
        parameter_loss = sum((p-q).square().mean() for p,q in zip(model.parameters(),original))
        loss = (stop_loss+args.moving_weight*move_loss+args.cap_weight*cap_loss
                +args.parameter_regularization*parameter_loss)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite offline optimization loss")
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % 100 == 0 or step == args.steps:
            score = training_score(model, dataset, args.moving_cap)
            current=copy.deepcopy(model.state_dict());blend_scores=[]
            for fraction in args.selection_blend_fractions:
                if fraction==1.:
                    candidate_state=current;candidate_score=score
                else:
                    candidate_state={key:initial_tail[key]+fraction*(current[key]-initial_tail[key]) for key in current}
                    model.load_state_dict(candidate_state)
                    candidate_score=training_score(model,dataset,args.moving_cap)
                blend_scores.append({"tail_fraction":fraction,**candidate_score})
                value=selection_value(candidate_score,initial,args)
                if value<best_score:
                    best_score,best_step,best_state,best_fraction=value,step,copy.deepcopy(candidate_state),fraction
            model.load_state_dict(current)
            history.append({"step": step, **score,"training_only_blends":blend_scores})
            print(json.dumps({"offline_step": step, "seconds": round(time.monotonic()-started,1),
                              "best_guarded_step": best_step,"best_tail_fraction":best_fraction, **score}), flush=True)
    if best_step == 0:
        raise ValueError("No nonlinear iterate improved the selected training objective while preserving required guards")
    model.load_state_dict(best_state)
    return model.export(state), {"algorithm": "Adam supervised offline target fitting", "optimization_steps": args.steps,
        "selected_step": best_step, "selection_objective":args.selection_objective,
        "selected_tail_fraction_from_warmstart":best_fraction,
        "tail_interpolation_scope":"Only existing final-two-layer tensors; selection on training rows with unchanged guards, never heldout.",
        "selection":("Lowest weighted training stop-target RMSE among training-guard-valid checkpoints" if args.selection_objective=="stop"
                     else "Lowest training stop MSE + moving_weight * weighted raw/applied moving-label MSE, with moving guards, strict moving-label loss improvement and bounded stop-RMSE regression"),
        "heldout_used_for_fit_selection_or_backoff": False, "prefix_replay_max_abs_error": replay,
        "initial_training": initial, "selected_training": training_score(model,dataset,args.moving_cap), "history": history}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,default=base.SOURCE)
    parser.add_argument("--traces",type=Path,default=base.TRACES)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--steps",type=int,default=2000)
    parser.add_argument("--batch-size",type=int,default=512)
    parser.add_argument("--learning-rate",type=float,default=.0005)
    parser.add_argument("--moving-weight",type=float,default=20.)
    parser.add_argument("--selection-objective","--selection",choices=("stop","joint"),default="stop")
    parser.add_argument("--stop-rmse-max-factor",type=float,default=1.15,
                        help="Joint-selection stop-RMSE bound relative to initialized tail on training rows")
    parser.add_argument("--selection-blend-fractions",type=float,nargs="+",default=[1.],
                        help="Training-only candidate tail interpolation fractions from initialized tail; default unblended")
    parser.add_argument("--cap-weight",type=float,default=100.)
    parser.add_argument("--moving-cap",type=float,default=.12)
    parser.add_argument("--parameter-regularization",type=float,default=.0001)
    parser.add_argument("--yaw-damping",type=float,default=0.)
    parser.add_argument("--vertical-damping",type=float,default=0.)
    parser.add_argument("--extra-stop-trace",action="append",default=[],metavar="PATH=PRODUCER_SHA256")
    parser.add_argument("--extra-stop-weight",type=float,default=4.)
    parser.add_argument("--extra-moving-trace",action="append",default=[],metavar="PATH=PRODUCER_SHA256")
    parser.add_argument("--extra-moving-weight",type=float,default=4.)
    parser.add_argument("--anchor-jitter",type=float,nargs=5,default=None,
                        metavar=("Q_STD","QDOT_STD","PREV_ACTION_STD","OMEGA_STD","V_STD"))
    parser.add_argument("--stop-anchor-weight",type=float,default=1.)
    parser.add_argument("--anchor-samples",type=int,default=256,
                        help="Samples per height only when --anchor-jitter is supplied; legacy defaults remain 32")
    parser.add_argument("--warmstart-tail",type=Path)
    parser.add_argument("--seed",type=int,default=999)
    args=parser.parse_args()
    if not 1<=args.steps<=2000 or args.batch_size<2 or args.batch_size%2:
        parser.error("Require 1..2000 offline steps and a positive even batch size")
    for key in ("learning_rate","moving_weight","cap_weight","moving_cap","parameter_regularization","extra_stop_weight","extra_moving_weight","stop_anchor_weight"):
        if not np.isfinite(getattr(args,key)) or getattr(args,key)<=0: parser.error(f"Invalid {key}")
    for key in ("yaw_damping","vertical_damping"):
        if not np.isfinite(getattr(args,key)) or getattr(args,key)<0: parser.error(f"Invalid {key}")
    if args.anchor_samples<2:parser.error("Require at least two synthetic anchor samples")
    if not np.isfinite(args.stop_rmse_max_factor) or args.stop_rmse_max_factor<1:
        parser.error("Stop RMSE maximum factor must be finite and at least one")
    if any(not np.isfinite(v) or not 0<v<=1 for v in args.selection_blend_fractions):
        parser.error("Tail interpolation fractions must be finite in (0,1]")
    if args.anchor_jitter is not None and any(not np.isfinite(v) or v<0 for v in args.anchor_jitter):
        parser.error("Anchor jitter standard deviations must be finite and nonnegative")
    if args.output.exists() or args.output.with_suffix(".json").exists(): parser.error("Refusing to overwrite artifacts")
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    if base.sha256(args.checkpoint)!=base.SOURCE_SHA: raise ValueError("Pinned source999 differs")
    content=torch.load(args.checkpoint,map_location="cpu",weights_only=True);state=content["actor_state_dict"]
    actor=base.FlatActor(state);urdf=base.ROOT/"src/RLforHexapod/assets/urdf/robot.urdf";kin=base.Kinematics(urdf)
    label_args=argparse.Namespace(yaw_damping=args.yaw_damping,vertical_damping=args.vertical_damping,stand_velocity_damping=.01,
                                  stop_strength=1.,stop_weight=1.,moving_weight=1.,seed=args.seed)
    anchors={};blocks,heldout,traces=base.load_recordings(args.traces,actor,kin,label_args,anchors)
    blocks.extend(synthetic_stop_neighborhoods(actor,kin,label_args,anchors,args.anchor_jitter,
                                               args.stop_anchor_weight,args.anchor_samples))
    extra_provenance=[]
    reference_report=json.loads((args.traces/"stand.json").read_text(encoding="utf-8"))
    for entry in args.extra_stop_trace:
        if "=" not in entry:parser.error("Extra stop format is PATH=PRODUCER_SHA256")
        path,producer=entry.rsplit("=",1)
        if len(producer)!=64 or any(c not in "0123456789abcdef" for c in producer):parser.error("Invalid producer SHA256")
        train_block,test_block,provenance=load_extra_stop(Path(path),producer,actor,kin,anchors,args.extra_stop_weight,reference_report)
        blocks.append(train_block);heldout.append(test_block);extra_provenance.append(provenance)
    extra_moving_provenance=[]
    for entry in args.extra_moving_trace:
        if "=" not in entry:parser.error("Extra moving format is PATH=PRODUCER_SHA256")
        path,producer=entry.rsplit("=",1)
        if len(producer)!=64 or any(c not in "0123456789abcdef" for c in producer):parser.error("Invalid producer SHA256")
        train_block,test_block,provenance=load_extra_moving(Path(path),producer,actor,args.extra_moving_weight,reference_report)
        blocks.append(train_block);heldout.append(test_block);extra_moving_provenance.append(provenance)
    initial_state,initialization=validated_initial_tail(args.warmstart_tail,state)
    edited_state,fit=optimize(state,blocks,args,initial_state)
    training,train_ok=base.evaluate_blocks(state,edited_state,blocks,args.moving_cap)
    validation,heldout_ok=base.evaluate_blocks(state,edited_state,heldout,args.moving_cap)
    if not train_ok or not heldout_ok:
        raise ValueError("Final heldout preservation failed; no candidate written: "+json.dumps(validation))
    changed=[k for k in state if not base.same(state[k],edited_state[k])]
    if not changed or not set(changed)<=set(EDIT_KEYS): raise AssertionError("Unexpected actor changes")
    for key in state:
        if key not in EDIT_KEYS and not base.same(state[key],edited_state[key]): raise AssertionError("Frozen actor tensor changed")
    edited=copy.deepcopy(content);edited["actor_state_dict"]=edited_state
    for key in content:
        if key!="actor_state_dict" and not base.same(content[key],edited[key]): raise AssertionError("Metadata changed")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("xb") as stream:torch.save(edited,stream)
    if not base.same(edited,torch.load(args.output,map_location="cpu",weights_only=True)): raise AssertionError("Serialization mismatch")
    if base.sha256(args.checkpoint)!=base.SOURCE_SHA:raise AssertionError("Source changed")
    anchor_rows=[kin.anchor(h)[1] for h in (.17,.205,.24)]
    report={"status":"experimental_nonlinear_weight_surgery_awaiting_physics", "method":"nonlinear",
        "edit_scope":"last_two_layers","prefix_preserved":True,"preserved_prefix_layers":[0,2],
        "upstream_preserved":True,"upstream_preserved_scope":"Frozen prefix layers 0 and 2; layer 4 is explicitly editable.",
        "source_sha256":base.SOURCE_SHA,"source_checkpoint":str(args.checkpoint.resolve()),
        "candidate":str(args.output.resolve()),"candidate_sha256":base.sha256(args.output),
        "source_sha256_unchanged":True,"normalization_epsilon":base.EPSILON,"normalizer_preserved":True,
        "changed_actor_tensors":changed,"architecture_unchanged":True,"runtime_inputs_added":False,
        "runtime_controller_added":False,"all_other_checkpoint_fields_preserved":True,"roundtrip_exact":True,
        "physics_validated":False,"generator_sha256":base.sha256(__file__),
        "target_helper_sha256":base.sha256(base.ROOT/"scripts/wsp_stop_refine.py"),"urdf_sha256":base.sha256(urdf),
        "settings":{k:v for k,v in vars(args).items() if not isinstance(v,Path)},
        "recorded_traces":traces,"extra_stop_recordings":extra_provenance,"initialization":initialization,
        "extra_moving_recordings":extra_moving_provenance,
        "synthetic_stop_neighborhood":{"enabled":args.anchor_jitter is not None,
            "jitter_order":["joint_position_rad","joint_velocity_rad_s","previous_applied_action","body_angular_velocity_rad_s","body_linear_velocity_m_s"],
            "standard_deviations":args.anchor_jitter,"samples_per_height":args.anchor_samples if args.anchor_jitter is not None else 32,
            "zero_command_only":True,"first_row_exact_equilibrium":True,"previous_action_clipped_to":[-1,1],
            "moving_protection_rows_unchanged":True,"physical_trajectories":False},
        "fit":fit,"training":training,"heldout":validation,
        "anchor_contract":{"heights":[.17,.205,.24],"joint_names":base.JOINT_NAMES,
            "joint_positions_rad":[a["joint_positions_rad"] for a in anchor_rows],"source_sha256":base.SOURCE_SHA,
            "physical_checked":False,"anchors":anchor_rows},
        "limitations":["Frozen-input guards do not guarantee speed or stability after the edited policy changes its state distribution.",
            "Offline target optimization is not physical reinforcement learning, and fitting static targets does not prove cessation of marching.",
            "Source prefix/normalizer remain exact, but changing the existing final hidden layer permits new nonlinear command sensitivity.",
            "Synthetic/counterfactual states are not physical trajectories; heldout samples are later blocks of the same source episodes.",
            "Optional yaw/vz corrections are offline rigid-contact kinematic labels; their real effect requires physical validation.",
            "Preserved optimizer/value states are stale; use actor-only warm-start with fresh optimizer/critic."]}
    with args.output.with_suffix(".json").open("x",encoding="utf-8") as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({"candidate":str(args.output),"candidate_sha256":report["candidate_sha256"],
                      "selected_step":fit["selected_step"],"heldout":validation}),flush=True)


if __name__=="__main__":main()
