"""계단 평가: 저장된 정책을 계단 앞에서 출발시켜 몇 %가 꼭대기까지 오르는지 잰다. 학습하지 않는다.

teacher와 student를 같은 조건으로 비교한다.
- 저장 버퍼를 쓰지 않는다. 0번 또는 1번 시작 위치에서만 출발한다.
- 노이즈 없는 평균 행동을 쓴다. student의 높이맵 노이즈는 그대로 둔다.
- 환경마다 첫 에피소드만 센다. 짧은 에피소드가 더 많이 집계되지 않는다.

    uv run python scripts/eval_stairs.py --task Hexapod-Stair --checkpoint <teacher.pt>
    uv run python scripts/eval_stairs.py --task Hexapod-Stair-Student-A --checkpoint <student.pt>
"""

import argparse
import contextlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

import torch

START_NAMES = {0: "0번 (계단 앞)", 1: "1번 (앞발이 수직면 앞)"}


class StairDiagnostics:
    """첫 에피소드의 정책 step별 상태. 접촉 링크는 충돌 후보이며 접촉 상대는 알 수 없다."""

    def __init__(self, stream, body_names, step_dt, max_envs=4, contact_threshold=1.0, metadata=None):
        if max_envs < 1:
            raise ValueError("max_envs must be positive")
        if not math.isfinite(contact_threshold) or contact_threshold < 0:
            raise ValueError("contact_threshold must be finite and nonnegative")
        self.stream = stream
        self.body_names = list(body_names)
        self.step_dt = step_dt
        self.max_envs = max_envs
        self.contact_threshold = contact_threshold
        self.seen_undesired = set()
        self.seen_tibia_horizontal = set()
        self._write({
            "type": "metadata",
            "run": metadata or {},
            "step_dt": step_dt,
            "max_envs": max_envs,
            "contact_threshold_n": contact_threshold,
            "contact_body_names": self.body_names,
            "sampling": (
                "pre-transition, pre-reset; strongest contact-history vector "
                "and maximum horizontal force per link"
            ),
            "contact_note": "Candidates only: no contact partner or exact first physics-substep ordering.",
        })

    def _write(self, record):
        # sim_nan으로 끝난 에피소드도 유효한 JSON으로 남긴다.
        def clean(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, list):
                return [clean(item) for item in value]
            if isinstance(value, dict):
                return {key: clean(item) for key, item in value.items()}
            return value

        self.stream.write(json.dumps(clean(record), ensure_ascii=False, allow_nan=False) + "\n")

    def record(self, step, snapshot, dones, finished, termination_manager):
        if snapshot is None:
            raise RuntimeError("StairTracker diagnostics were not captured before reset")
        count = min(self.max_envs, len(finished))
        active = (~finished[:count]).cpu().tolist()
        if not any(active):
            return
        values = {key: value[:count].detach().cpu().tolist() for key, value in snapshot.items()}
        done = dones[:count].cpu().tolist()
        reasons = {
            name: termination_manager.get_term(name)[:count].cpu().tolist()
            for name in termination_manager.active_terms
        }
        for env_id, keep in enumerate(active):
            if not keep:
                continue
            sample = {key: value[env_id] for key, value in values.items()}
            forces = sample["contact_forces_w"]
            horizontal = sample["horizontal_contact_force"]
            undesired, tibia_horizontal = [], []
            for name, force, horizontal_force in zip(self.body_names, forces, horizontal, strict=True):
                if name == "body" or name.startswith(("coxa", "femur")):
                    if math.sqrt(sum(component * component for component in force)) > self.contact_threshold:
                        undesired.append(name)
                elif name.startswith("tibia") and horizontal_force > self.contact_threshold:
                    tibia_horizontal.append(name)
            first_undesired = undesired if env_id not in self.seen_undesired else []
            first_horizontal = tibia_horizontal if env_id not in self.seen_tibia_horizontal else []
            if undesired:
                self.seen_undesired.add(env_id)
            if tibia_horizontal:
                self.seen_tibia_horizontal.add(env_id)
            self._write({
                "type": "step",
                "step": step,
                "time_s": step * self.step_dt,
                "env_id": env_id,
                **sample,
                "episode_done": done[env_id],
                "termination_terms": [name for name, flags in reasons.items() if flags[env_id]],
                "undesired_contact_links": undesired,
                "tibia_horizontal_contact_links": tibia_horizontal,
                "first_sampled_undesired_contact_links": first_undesired,
                "first_sampled_tibia_horizontal_contact_links": first_horizontal,
            })
        self.stream.flush()


def collect_first_episodes(env, policy, obs, tracker, termination_manager, max_steps, diagnostics=None):
    """환경마다 첫 에피소드가 끝날 때까지 돌리고 결과를 모은다."""
    term_names = termination_manager.active_terms
    start = tracker.move.clone()
    ended = torch.zeros_like(start)
    reasons = torch.zeros(start.shape[0], len(term_names), dtype=torch.bool, device=start.device)
    finished = torch.zeros_like(start, dtype=torch.bool)
    length = torch.zeros_like(start)  # 에피소드 길이 [step]
    top_step = torch.full_like(start, -1)  # 꼭대기에 처음 도달한 step. -1 = 도달 못 함

    with torch.inference_mode():
        for step in range(1, max_steps + 1):
            obs, _, dones, _ = env.step(policy(obs))
            dones = dones.bool()
            policy.reset(dones)  # student의 기억을 에피소드마다 비운다

            if diagnostics is not None:
                # tracker.update()의 복사본은 자동 리셋 뒤에도 마지막 step의 상태를 보존한다.
                diagnostics.record(step, tracker.diagnostics, dones, finished, termination_manager)

            # step 안에서 리셋까지 끝났으므로, 끝난 환경의 tracker.move는 이미 다음 에피소드 값이다
            move = torch.where(dones, tracker.ended_move, tracker.move)
            top_step[(move >= tracker.final_advance) & (top_step < 0) & ~finished] = step

            newly = dones & ~finished
            ended[newly] = tracker.ended_move[newly]
            reasons[newly] = torch.stack([termination_manager.get_term(n) for n in term_names], dim=1)[newly]
            length[newly] = step
            finished |= newly
            if finished.all():
                break

    if not finished.all():
        raise RuntimeError(f"{max_steps} step 안에 끝나지 않은 환경이 {(~finished).sum().item()}개 있습니다.")
    return {
        "start": start,
        "ended": ended,
        "reasons": reasons,
        "term_names": term_names,
        "length": length,
        "top_step": top_step,
    }


def summarize(data, mask, final_advance, done_move, step_dt):
    """mask로 고른 에피소드들의 결과."""
    ended = data["ended"][mask]
    length = data["length"][mask]
    top_step = data["top_step"][mask]
    count = len(ended)
    reached_top = (ended >= final_advance).float().mean().item()
    result = {
        "count": count,
        "reached_top": reached_top,
        "margin": 1.96 * (reached_top * (1.0 - reached_top) / count) ** 0.5,  # 95% 신뢰구간 반폭
        "success": (ended == done_move).float().mean().item(),
        "mean_move": ended.float().mean().item(),
        "move_counts": torch.bincount(ended, minlength=done_move + 1).tolist(),
        "reasons": dict(zip(data["term_names"], data["reasons"][mask].float().mean(dim=0).tolist(), strict=True)),
        "top_time": None,
        "top_share": 0.0,
    }
    reached = top_step >= 0
    if reached.any():
        seconds = top_step[reached].float() * step_dt
        result["top_time"] = (seconds.mean().item(), seconds.median().item(), seconds.quantile(0.9).item())
        # 전체 시뮬레이션 시간 중 꼭대기에 도달한 뒤의 비중
        result["top_share"] = ((length[reached] - top_step[reached]).sum() / length.sum()).item()
    return result


def format_report(data, final_advance, done_move, step_dt):
    """전체와 시작 위치별 결과를 글로 만든다."""
    start = data["start"]
    lines = [f"꼭대기 도달 = 단계 {final_advance} 이상, 정지 성공 = 단계 {done_move}"]
    groups = [("전체", torch.ones_like(start, dtype=torch.bool))]
    for value in start.unique().tolist():
        groups.append((START_NAMES.get(value, f"{value}번"), start == value))

    for name, mask in groups:
        result = summarize(data, mask, final_advance, done_move, step_dt)
        reasons_text = ", ".join(f"{k} {v:.1%}" for k, v in result["reasons"].items() if v > 0)
        moves_text = " ".join(f"{k}:{v}" for k, v in enumerate(result["move_counts"]) if v > 0)
        if result["top_time"] is None:
            time_text = "도달한 로봇 없음"
        else:
            time_text = "평균 {:.1f}초, 중앙값 {:.1f}초, 90%가 {:.1f}초 안에 도달".format(*result["top_time"])
        lines += [
            "",
            f"[{name}] {result['count']}대",
            f"  꼭대기 도달: {result['reached_top']:.1%} (±{result['margin']:.1%})",
            f"  정지 성공:   {result['success']:.1%}",
            f"  평균 단계:   {result['mean_move']:.2f} / {done_move}",
            f"  끝난 단계:   {moves_text}",
            f"  종료 이유:   {reasons_text}",
            f"  도달 시간:   {time_text}",
            f"  꼭대기 체류: 전체 시간의 {result['top_share']:.1%}",
        ]
    return "\n".join(lines)


def main():
    import gymnasium as gym

    from isaaclab.app import add_launcher_args, launch_simulation

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

    import RLforHexapod.tasks  # noqa: F401

    parser = argparse.ArgumentParser(description="저장된 정책의 계단 오르기 평가")
    parser.add_argument("--task", required=True, help="Hexapod-Stair 또는 Hexapod-Stair-Student-*")
    parser.add_argument("--checkpoint", required=True, help="불러올 .pt 파일")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diagnostics", type=Path, help="충돌 원인 분석용 JSONL 저장 경로 (새 파일)")
    parser.add_argument("--diagnostic-envs", type=int, default=4, help="앞에서부터 기록할 환경 수")
    parser.add_argument("--contact-threshold", type=float, default=1.0, help="접촉 후보 검출 힘 [N]")
    add_launcher_args(parser)
    args, remaining = setup_preset_cli(parser, sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    if args.diagnostic_envs < 1:
        parser.error("--diagnostic-envs must be positive")
    if not math.isfinite(args.contact_threshold) or args.contact_threshold < 0:
        parser.error("--contact-threshold must be finite and nonnegative")

    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")

    with launch_simulation(env_cfg, args), contextlib.ExitStack() as cleanup:
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = agent_cfg.seed = args.seed
        if getattr(args, "device", None) is not None:
            env_cfg.sim.device = agent_cfg.device = args.device
        env_cfg.events.stair_reset.params["buffer_ratio"] = 0.0

        env = gym.make(args.task, cfg=env_cfg)
        cleanup.callback(env.close)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)  # 여기서 첫 리셋이 일어난다

        runner = create_rsl_rl_runner(env, agent_cfg)
        print(f"[INFO] 체크포인트: {args.checkpoint}")
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        tracker = env.unwrapped.stair_tracker
        diagnostics = None
        if args.diagnostics is not None:
            args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
            stream = cleanup.enter_context(args.diagnostics.open("x", encoding="utf-8"))
            diagnostics = StairDiagnostics(
                stream, tracker.sensor.body_names, env.unwrapped.step_dt,
                max_envs=args.diagnostic_envs, contact_threshold=args.contact_threshold,
                metadata={
                    "task": args.task, "checkpoint": args.checkpoint, "seed": args.seed,
                    "num_envs": args.num_envs,
                    "move_contact_hold_time_s": tracker.gait.move_contact_hold_time,
                },
            )
            tracker.diagnostics_enabled = True
        data = collect_first_episodes(
            env,
            policy,
            env.get_observations(),
            tracker,
            env.unwrapped.termination_manager,
            env.unwrapped.max_episode_length + 1,
            diagnostics=diagnostics,
        )
        if diagnostics is not None:
            print(f"[INFO] 단계·접촉 진단: {args.diagnostics}")
        print(f"\n===== {args.task} =====")
        print(format_report(data, tracker.final_advance, tracker.done_move, env.unwrapped.step_dt))


if __name__ == "__main__":
    main()
