"""키보드 조종기: 누를 때마다 명령을 한 칸씩 바꾸고, HXCMD v1으로 50 Hz 보낸다. 표준 라이브러리만 쓴다.

이 콘솔 창에 포커스가 있을 때만 키를 읽는다 (뷰어 창의 카메라 조작 키와 섞이지 않는다).
명령은 다시 바꿀 때까지 유지된다. 이 프로그램을 끄면 받는 쪽이 0.25초 뒤 스스로 멈춘다.

    uv run python scripts/teleop/command_sender.py                 # 같은 PC의 시뮬레이션
    uv run python scripts/teleop/command_sender.py --host 192.168.0.10
"""

import argparse
import math
import os
import sys
import time

from protocol import CMD_HZ, PORT, STATUS_CLAMPED, STATUS_FALLEN, STATUS_TIMEOUT, CommandSender, Mode

HELP = """\
  W / S   앞 / 뒤       (vx)        A / D   왼쪽 / 오른쪽 (vy)
  Q / E   왼쪽 / 오른쪽 회전 (wz)    R / F   몸 높이 올림 / 내림
  Space   속도 0        H   높이 기본값      T   리셋      X   비상정지 켜기/끄기
  Esc     종료"""

# 조종기 쪽 느슨한 한계. 실제 학습 범위로 자르는 건 받는 쪽이고, 잘리면 STATE를 보고 그 값으로 맞춘다
LOOSE = {"vx": (-1.0, 1.0), "vy": (-1.0, 1.0), "wz": (-2.0, 2.0), "height": (0.10, 0.30)}
STATE_STALE = 1.0  # 이 시간 동안 STATE가 없으면 "응답 없음" [s]


class KeyReader:
    """눌린 키를 블로킹 없이 읽는다. Windows는 msvcrt, 그 밖에는 termios."""

    def __enter__(self):
        if os.name == "nt":
            import msvcrt
            self._msvcrt = msvcrt
        else:
            import termios
            import tty
            self._termios = termios
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc):
        if os.name != "nt":
            self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, self._old)

    def read(self) -> list[str]:
        out = []
        if os.name == "nt":
            while self._msvcrt.kbhit():
                ch = self._msvcrt.getwch()
                if ch in ("\x00", "\xe0"):  # 방향키 등 특수키: 두 번째 글자를 버린다
                    self._msvcrt.getwch()
                    continue
                out.append(ch)
        else:
            import select
            # sys.stdin.read는 파이썬 버퍼에 미리 읽어 select와 어긋나므로 os.read로 직접 읽는다
            data = ""
            while select.select([self._fd], [], [], 0.0)[0]:
                data += os.read(self._fd, 1024).decode(errors="ignore")
            i = 0
            while i < len(data):
                if data[i] == "\x1b" and i + 1 < len(data):  # 방향키 등 (ESC [ A): 끝 글자까지 버린다
                    i += 2
                    while i < len(data) and not (data[i].isalpha() or data[i] == "~"):
                        i += 1
                else:
                    out.append(data[i])
                i += 1
        return out


def fmt(v: float, sign: bool = True) -> str:
    if math.isnan(v):
        return "  --  "
    return f"{v:+.2f}" if sign else f"{v:.3f}"


def main():
    parser = argparse.ArgumentParser(description="키보드로 HXCMD v1 명령 보내기")
    parser.add_argument("--host", default="127.0.0.1", help="시뮬레이션/로봇 IP")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--step_v", type=float, default=0.05, help="W/S/A/D 한 번에 바뀌는 속도 [m/s]")
    parser.add_argument("--step_w", type=float, default=0.1, help="Q/E 한 번에 바뀌는 회전 속도 [rad/s]")
    parser.add_argument("--step_h", type=float, default=0.01, help="R/F 한 번에 바뀌는 높이 [m]")
    parser.add_argument("--height", type=float, default=0.20, help="높이 기본값 [m]")
    args = parser.parse_args()

    tx = CommandSender(args.host, args.port)
    cmd = {"vx": 0.0, "vy": 0.0, "wz": 0.0, "height": args.height}
    steps = {"w": ("vx", args.step_v), "s": ("vx", -args.step_v), "a": ("vy", args.step_v),
             "d": ("vy", -args.step_v), "q": ("wz", args.step_w), "e": ("wz", -args.step_w),
             "r": ("height", args.step_h), "f": ("height", -args.step_h)}
    estop = False
    last_state_seq = None
    last_state_time = -math.inf

    print(f"HXCMD v1 → {args.host}:{args.port}\n{HELP}\n")
    period = 1.0 / CMD_HZ
    next_time = time.perf_counter()
    frame = 0
    try:
        with KeyReader() as kr:
            while True:
                for ch in kr.read():
                    k = ch.lower()
                    if k in steps:
                        name, d = steps[k]
                        lo, hi = LOOSE[name]
                        cmd[name] = round(min(max(cmd[name] + d, lo), hi), 4)
                    elif k == " ":
                        cmd["vx"] = cmd["vy"] = cmd["wz"] = 0.0
                    elif k == "h":
                        cmd["height"] = args.height
                    elif k == "t":
                        tx.request_reset()
                    elif k == "x":
                        estop = not estop
                    elif ch == "\x1b":
                        raise KeyboardInterrupt

                tx.send(cmd["vx"], cmd["vy"], cmd["wz"], cmd["height"], Mode.POLICY, estop)

                st = tx.poll_state()
                now = time.perf_counter()
                if st is not None and st.seq != last_state_seq:
                    last_state_seq = st.seq
                    last_state_time = now
                    # 받는 쪽이 범위로 잘랐으면 그 값으로 맞춘다. 안 그러면 키를 더 눌러도 안 바뀌는 칸이 쌓인다
                    if st.status & STATUS_CLAMPED and st.mode == Mode.POLICY and not st.status & STATUS_TIMEOUT:
                        cmd.update(zip(("vx", "vy", "wz", "height"), (round(x, 4) for x in st.cmd), strict=True))

                frame += 1
                if frame % 5 == 0:  # 화면은 10 Hz로 갱신
                    line = (f"명령 vx {fmt(cmd['vx'])} vy {fmt(cmd['vy'])} wz {fmt(cmd['wz'])}"
                            f" h {fmt(cmd['height'], False)}")
                    if now - last_state_time > STATE_STALE:
                        line += " | 응답 없음"
                    else:
                        m = st.meas
                        line += (f" | 측정 vx {fmt(m[0])} vy {fmt(m[1])} wz {fmt(m[2])} h {fmt(m[3], False)}"
                                 f" | {tx.rtt_ms}ms")
                        flags = [s for bit, s in ((STATUS_TIMEOUT, "끊김"), (STATUS_CLAMPED, "잘림"),
                                                  (STATUS_FALLEN, "넘어짐")) if st.status & bit]
                        if flags:
                            line += " " + ",".join(flags)
                    if estop:
                        line += " [비상정지]"
                    print("\r" + line.ljust(110), end="", flush=True)

                next_time += period
                wait = next_time - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                else:
                    next_time = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        # 끝낼 때 정지 명령을 몇 번 보낸다. 사라져도 받는 쪽이 0.25초 뒤 타임아웃으로 멈춘다
        for _ in range(5):
            tx.send(0.0, 0.0, 0.0, cmd["height"], Mode.STOP, estop)
            time.sleep(period)
        tx.close()
        print("\n종료")


if __name__ == "__main__":
    main()
