"""시뮬레이션 조종: 학습된 걷기 정책을 env 1개로 띄우고, UDP(HXCMD v1)로 받은 명령을 따르게 한다.

명령 텐서를 직접 덮어쓴다. 속도와 높이 명령 항목의 _update_command를 바꿔 끼워,
매 step 명령 계산의 마지막에 조종기 값이 들어가게 한다. 그래서 다음 경우에도 조종기 값이 유지된다.
- 10초마다 다시 뽑기 (resampling)
- 정지 명령 env (rel_standing_envs)
- 넘어져서 리셋된 직후

    # 터미널 1: 시뮬레이션 (체크포인트를 안 주면 logs/rsl_rl/hexapod_walk의 최신 것)
    uv run python scripts/teleop/teleop.py --viz newton_gl --checkpoint <model.pt>
    # 터미널 2: 키보드 조종기
    uv run python scripts/teleop/command_sender.py
"""

import argparse
import contextlib
import importlib.metadata
import os
import sys
import time

import torch
from protocol import PORT, STATE_HZ, CommandReceiver, Limits

FALL_REPORT_TIME = 1.0  # 넘어진 뒤 STATE에 넘어짐 표시를 유지하는 시간 [s]


def override_commands(env, vel_target: torch.Tensor, height_target: torch.Tensor):
    """속도·높이 명령이 매 step 끝에 vel_target, height_target이 되게 한다. 두 텐서를 바꾸면 명령이 바뀐다."""
    cm = env.command_manager
    vel = cm.get_term("base_velocity")
    height = cm.get_term("base_height")
    # CommandTerm.compute()는 다시 뽑기 → _update_command 순서다. 마지막 단계를 바꾸면 무엇을 뽑았든 덮어쓴다
    vel._update_command = lambda: vel.vel_command_b.copy_(vel_target.expand_as(vel.vel_command_b))
    height._update_command = lambda: height.height_command.copy_(height_target.expand_as(height.height_command))
    # 처음 관측은 RslRlVecEnvWrapper가 리셋할 때 무작위 명령으로 이미 계산됐다. 지금 값으로 다시 계산한다
    vel._update_command()
    height._update_command()
    env.obs_buf = env.observation_manager.compute()


def measure(env, base_height) -> list[float]:
    """측정값 vx, vy [m/s], wz [rad/s], 몸통 높이 [m] (env 0)."""
    robot = env.scene["robot"]
    v = robot.data.root_lin_vel_b.torch[0]
    w = robot.data.root_ang_vel_b.torch[0]
    h = base_height(env, robot)[0]
    return torch.stack((v[0], v[1], w[2], h)).tolist()


def main():
    import gymnasium as gym

    from isaaclab.app import add_launcher_args, launch_simulation

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config, setup_preset_cli

    import RLforHexapod.tasks  # noqa: F401
    from RLforHexapod.tasks.walk.mdp import base_height

    parser = argparse.ArgumentParser(description="UDP 명령으로 걷기 정책 조종")
    parser.add_argument("--task", default="Hexapod-Walk", help="속도·높이 명령을 관측하는 task")
    parser.add_argument("--checkpoint", default=None, help="불러올 .pt 파일. 없으면 최신 run의 최신 체크포인트")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--default_height", type=float, default=0.20, help="조종기가 연결되기 전 몸통 높이 [m]")
    add_launcher_args(parser)
    args, remaining = setup_preset_cli(parser, sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining

    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point")

    with launch_simulation(env_cfg, args), contextlib.ExitStack() as cleanup:
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 1.0e6  # 시간 초과로 리셋되지 않게 한다. 넘어지면 리셋된다
        if getattr(args, "device", None) is not None:
            env_cfg.sim.device = agent_cfg.device = args.device

        ranges = env_cfg.commands.base_velocity.ranges
        limits = Limits(vx=ranges.lin_vel_x, vy=ranges.lin_vel_y, wz=ranges.ang_vel_z,
                        height=env_cfg.commands.base_height.height_range)
        print(f"[INFO] 명령 범위: {limits}")

        if args.checkpoint:
            checkpoint = os.path.abspath(args.checkpoint)
        else:
            log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
            checkpoint = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] 체크포인트: {checkpoint}")

        env = gym.make(args.task, cfg=env_cfg)
        cleanup.callback(env.close)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = create_rsl_rl_runner(env, agent_cfg)
        runner.load(checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        base = env.unwrapped

        rx = CommandReceiver(limits, default_height=args.default_height, port=args.port)
        cleanup.callback(rx.close)
        print(f"[INFO] UDP {args.port}번 포트에서 명령을 기다립니다. 조종기가 없으면 제자리에 섭니다. Ctrl+C로 종료")

        dt = base.step_dt
        state_every = max(1, round(1.0 / (dt * STATE_HZ)))
        with torch.inference_mode():
            vel_target = torch.zeros(1, 3, device=base.device)
            height_target = torch.full((1, 1), rx.height, device=base.device)
            override_commands(base, vel_target, height_target)
            obs = env.get_observations()

            step = 0
            connected = False
            fallen_until = 0.0
            next_time = time.perf_counter()
            late_steps = 0
            with contextlib.suppress(KeyboardInterrupt):
                while base.sim.is_running():  # 뷰어 창을 닫으면 끝난다
                    a = rx.poll()
                    if connected != (not a.timed_out):
                        connected = not a.timed_out
                        print(f"[INFO] 조종기 {'연결됨 ' + str(rx.addr) if connected else '끊김: 속도 0, 높이 유지'}")
                    vel_target.copy_(torch.tensor([[a.vx, a.vy, a.wz]]))
                    height_target.fill_(a.height)
                    if a.reset_requested:
                        base.sim.request_reset()  # 다음 env.step 안에서 리셋하고, 명령과 관측도 다시 계산한다
                        print("[INFO] 리셋 요청")

                    obs, _, dones, _ = env.step(policy(obs))
                    policy.reset(dones)
                    if dones.any() and not a.reset_requested:
                        fallen_until = time.perf_counter() + FALL_REPORT_TIME
                        print("[INFO] 넘어져서 리셋됨")

                    step += 1
                    if step % state_every == 0:
                        rx.send_state(measure(base, base_height), fallen=time.perf_counter() < fallen_until)

                    # 실시간에 맞춘다. 시뮬레이션이 실시간보다 느리면 따라잡으려 하지 않고 기준을 다시 잡는다
                    next_time += dt
                    wait = next_time - time.perf_counter()
                    if wait > 0:
                        time.sleep(wait)
                    elif wait < -dt:
                        next_time = time.perf_counter()
                        late_steps += 1
                        if late_steps % 50 == 1:
                            print(f"[WARN] 시뮬레이션이 실시간보다 느립니다 (늦은 step {late_steps}회)")


if __name__ == "__main__":
    main()
