"""HXCMD v1: 조종기 ↔ 시뮬레이션/실물 로봇 명령 프로토콜 (UDP).

표준 라이브러리만 쓴다 (torch, Isaac Lab 없음). 시뮬레이션(teleop.py), 조종기(command_sender.py),
나중의 라즈베리파이가 이 파일 하나를 같이 쓴다.

CMD   조종기 → 로봇, 50 Hz, 32바이트
    magic "HX" | version u8 | type u8 (1) | seq u32 | t_ms u32 | mode u8 | flags u8 | reset_count u8 | 0 u8
    | vx f32 [m/s] | vy f32 [m/s] | wz f32 [rad/s] | height f32 [m]
STATE 로봇 → 조종기, 약 20 Hz, 52바이트
    magic "HX" | version u8 | type u8 (2) | seq u32 | echo_seq u32 | echo_t_ms u32 | status u8 | mode u8 | 0 0
    | 적용한 명령 vx vy wz h (f32×4) | 측정값 vx vy wz h (f32×4, 모르면 NaN)

좌표: 몸체 좌표계. +x = 라이다 쪽(앞), +y = 왼쪽, wz는 위에서 볼 때 반시계가 +.
height: 몸통 원점의 지면 기준 높이.

받는 쪽 규칙 (CommandReceiver가 처리한다):
  1. 크기·magic·version·type이 틀리거나 값에 NaN/inf가 있으면 버린다.
  2. seq가 마지막보다 크지 않으면 버린다 (Wi-Fi는 순서가 뒤바뀔 수 있다). 타임아웃 중에는 받아들인다.
  3. 명령은 학습 범위로 자른다.
  4. CMD_TIMEOUT 동안 정상 패킷이 없으면 속도 0 (높이는 유지).
  5. 처음 보낸 조종기 주소만 따른다. 다른 주소는 그 조종기가 타임아웃된 뒤에 받아들인다.
  6. reset_count가 바뀌면 리셋을 한 번 요청한다 (패킷 하나가 사라져도 다음 패킷에 바뀐 값이 실린다).
"""

from __future__ import annotations

import contextlib
import math
import socket
import struct
import time
from dataclasses import dataclass
from enum import IntEnum

MAGIC = b"HX"
VERSION = 1
TYPE_CMD = 1
TYPE_STATE = 2

PORT = 15500
CMD_HZ = 50.0
STATE_HZ = 20.0
CMD_TIMEOUT = 0.25  # [s]

_CMD = struct.Struct("<2sBBIIBBBx4f")
_STATE = struct.Struct("<2sBBIIIBB2x4f4f")
assert _CMD.size == 32 and _STATE.size == 52

U32 = 1 << 32


class Mode(IntEnum):
    STOP = 0    # 속도 0 강제 (높이는 따른다)
    POLICY = 1  # 정책이 명령을 따른다
    # 2 이상은 접기·계단 모드 등을 위해 비워 둔다


FLAG_ESTOP = 1 << 0           # CMD flags: 비상정지

STATUS_TIMEOUT = 1 << 0       # STATE status: 타임아웃 중
STATUS_CLAMPED = 1 << 1       # 명령이 범위 밖이라 잘림
STATUS_FALLEN = 1 << 2        # 넘어짐 또는 리셋됨


def now_ms() -> int:
    """보내는 쪽 시계 [ms], u32로 감긴다. 왕복 지연 계산에만 쓰므로 양쪽 시계를 맞출 필요가 없다."""
    return int(time.monotonic() * 1000.0) % U32


def seq_newer(a: int, b: int) -> bool:
    """u32 seq a가 b보다 나중인가 (감김 고려)."""
    d = (a - b) % U32
    return 0 < d < (U32 >> 1)


def _finite(*xs: float) -> bool:
    return all(math.isfinite(x) for x in xs)


@dataclass
class Command:
    seq: int = 0
    t_ms: int = 0
    mode: int = Mode.STOP
    flags: int = 0
    reset_count: int = 0
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    height: float = 0.0

    @property
    def estop(self) -> bool:
        return bool(self.flags & FLAG_ESTOP)

    def pack(self) -> bytes:
        return _CMD.pack(MAGIC, VERSION, TYPE_CMD, self.seq % U32, self.t_ms % U32,
                         int(self.mode), self.flags, self.reset_count % 256,
                         self.vx, self.vy, self.wz, self.height)

    @classmethod
    def unpack(cls, data: bytes) -> Command | None:
        """형식이 틀리거나 값이 유한하지 않으면 None."""
        if len(data) != _CMD.size:
            return None
        magic, ver, typ, seq, t_ms, mode, flags, rc, vx, vy, wz, h = _CMD.unpack(data)
        if magic != MAGIC or ver != VERSION or typ != TYPE_CMD or not _finite(vx, vy, wz, h):
            return None
        return cls(seq, t_ms, mode, flags, rc, vx, vy, wz, h)


@dataclass
class State:
    seq: int = 0
    echo_seq: int = 0
    echo_t_ms: int = 0
    status: int = 0
    mode: int = Mode.STOP
    cmd: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)   # 적용한 명령 vx vy wz h
    meas: tuple[float, float, float, float] = (math.nan,) * 4        # 측정값 vx vy wz h, 모르면 NaN

    def pack(self) -> bytes:
        return _STATE.pack(MAGIC, VERSION, TYPE_STATE, self.seq % U32, self.echo_seq % U32,
                           self.echo_t_ms % U32, self.status, int(self.mode), *self.cmd, *self.meas)

    @classmethod
    def unpack(cls, data: bytes) -> State | None:
        if len(data) != _STATE.size:
            return None
        v = _STATE.unpack(data)
        if v[0] != MAGIC or v[1] != VERSION or v[2] != TYPE_STATE:
            return None
        return cls(v[3], v[4], v[5], v[6], v[7], tuple(v[8:12]), tuple(v[12:16]))


@dataclass
class Limits:
    """학습 범위. 받는 쪽은 명령을 여기로 자른다.
    teleop.py는 env cfg에서 읽어 채우므로, 기본값은 env cfg가 없는 쪽(실물 로봇 등)에서만 쓰인다."""
    vx: tuple[float, float] = (-0.2, 0.2)
    vy: tuple[float, float] = (-0.2, 0.2)
    wz: tuple[float, float] = (-0.5, 0.5)
    height: tuple[float, float] = (0.17, 0.24)

    def clamp(self, vx, vy, wz, h) -> tuple[tuple[float, float, float, float], bool]:
        out = (min(max(vx, self.vx[0]), self.vx[1]), min(max(vy, self.vy[0]), self.vy[1]),
               min(max(wz, self.wz[0]), self.wz[1]), min(max(h, self.height[0]), self.height[1]))
        # f32로 오면 경계값(예: 0.2)이 0.2000000029가 되므로 작은 차이는 잘림으로 치지 않는다
        return out, any(abs(o - i) > 1e-5 for o, i in zip(out, (vx, vy, wz, h), strict=True))


@dataclass
class Applied:
    """받는 쪽이 이번 step에 실제로 쓸 명령."""
    vx: float
    vy: float
    wz: float
    height: float
    mode: int
    estop: bool
    timed_out: bool
    clamped: bool
    reset_requested: bool  # 이번 poll에서 reset_count가 바뀌었다 (한 번만 True)


def _udp_socket() -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    # Windows: 상대가 꺼진 뒤 sendto하면 다음 recvfrom이 ConnectionResetError를 낸다. 끈다.
    if hasattr(socket, "SIO_UDP_CONNRESET"):
        sock.ioctl(socket.SIO_UDP_CONNRESET, False)
    return sock


def _recv_all(sock: socket.socket, max_packets: int = 256):
    """쌓인 패킷을 모두 꺼낸다 (블로킹 없음)."""
    for _ in range(max_packets):
        try:
            data, addr = sock.recvfrom(256)
        except (BlockingIOError, InterruptedError):
            return
        except (ConnectionResetError, ConnectionRefusedError):
            continue
        yield data, addr


class CommandReceiver:
    """로봇/시뮬레이션 쪽. 매 제어 step마다 poll()을 부르고, 가끔 send_state()로 상태를 돌려보낸다."""

    def __init__(self, limits: Limits | None = None, default_height: float | None = None,
                 port: int = PORT, host: str = "0.0.0.0", timeout: float = CMD_TIMEOUT):
        self.limits = limits or Limits()
        self.timeout = timeout
        self.sock = _udp_socket()
        self.sock.bind((host, port))
        h = default_height if default_height is not None else sum(self.limits.height) / 2.0
        self.height = self.limits.clamp(0.0, 0.0, 0.0, h)[0][3]  # 타임아웃 때 유지할 높이
        self.last: Command | None = None   # 마지막으로 받아들인 CMD
        self.addr = None                   # 따르는 조종기 주소
        self._last_time = -math.inf        # 마지막 정상 패킷 시각 (monotonic)
        self._state_seq = 0
        self._applied: Applied | None = None

    def timed_out(self, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return now - self._last_time > self.timeout

    def _accept(self, cmd: Command, addr, now: float) -> bool:
        if self.addr is None or self.timed_out(now):
            # 새 조종기 (또는 재시작한 조종기): seq를 처음부터 받아들인다
            self.addr = addr
            return True
        return addr == self.addr and seq_newer(cmd.seq, self.last.seq)

    def poll(self) -> Applied:
        now = time.monotonic()
        reset = False
        for data, addr in _recv_all(self.sock):
            cmd = Command.unpack(data)
            if cmd is None:
                continue
            same_sender = self.last is not None and addr == self.addr and not self.timed_out(now)
            if not self._accept(cmd, addr, now):
                continue
            # 같은 조종기에서 reset_count가 바뀌었을 때만 리셋 (새 조종기의 첫 값은 기준으로만 쓴다)
            if same_sender and cmd.reset_count != self.last.reset_count:
                reset = True
            self.last = cmd
            self._last_time = now

        timed_out = self.timed_out(now)
        if timed_out or self.last is None:
            a = Applied(0.0, 0.0, 0.0, self.height, Mode.STOP, False, True, False, reset)
        else:
            c = self.last
            (vx, vy, wz, h), clamped = self.limits.clamp(c.vx, c.vy, c.wz, c.height)
            self.height = h
            mode = Mode.STOP if c.estop else c.mode
            if mode != Mode.POLICY:  # STOP과 아직 정하지 않은 모드는 속도 0
                vx = vy = wz = 0.0
            a = Applied(vx, vy, wz, h, mode, c.estop, False, clamped, reset)
        self._applied = a
        return a

    def send_state(self, meas: tuple[float, float, float, float] = (math.nan,) * 4, fallen: bool = False):
        """마지막 CMD를 보낸 주소로 STATE를 보낸다. 받은 적이 없으면 보내지 않는다."""
        if self.addr is None or self._applied is None:
            return
        a = self._applied
        status = (STATUS_TIMEOUT * a.timed_out) | (STATUS_CLAMPED * a.clamped) | (STATUS_FALLEN * fallen)
        st = State(self._state_seq, self.last.seq, self.last.t_ms, status, a.mode,
                   (a.vx, a.vy, a.wz, a.height), tuple(float(x) for x in meas))
        self._state_seq = (self._state_seq + 1) % U32
        with contextlib.suppress(OSError):
            self.sock.sendto(st.pack(), self.addr)

    def close(self):
        self.sock.close()


class CommandSender:
    """조종기 쪽. send()를 CMD_HZ로 부르고, poll_state()로 최신 STATE와 왕복 지연을 받는다."""

    def __init__(self, host: str, port: int = PORT):
        self.target = (host, port)
        self.sock = _udp_socket()
        self.seq = 0
        self.reset_count = 0
        self.state: State | None = None
        self.rtt_ms: int | None = None

    def request_reset(self):
        self.reset_count = (self.reset_count + 1) % 256

    def send(self, vx: float, vy: float, wz: float, height: float,
             mode: int = Mode.POLICY, estop: bool = False):
        cmd = Command(self.seq, now_ms(), mode, FLAG_ESTOP if estop else 0, self.reset_count,
                      vx, vy, wz, height)
        self.seq = (self.seq + 1) % U32
        # 네트워크가 잠깐 끊겨도 계속 보낸다. 받는 쪽이 타임아웃으로 멈춘다
        with contextlib.suppress(OSError):
            self.sock.sendto(cmd.pack(), self.target)

    def poll_state(self) -> State | None:
        """새로 받은 STATE 중 가장 최신 것을 self.state에 넣고 돌려준다 (없으면 이전 값)."""
        for data, _ in _recv_all(self.sock):
            st = State.unpack(data)
            if st is None or (self.state is not None and not seq_newer(st.seq, self.state.seq)):
                continue
            self.state = st
            self.rtt_ms = (now_ms() - st.echo_t_ms) % U32
        return self.state

    def close(self):
        self.sock.close()


def _selftest():
    """localhost에서 규칙을 확인한다: python scripts/teleop/protocol.py"""
    port = 15599
    rx = CommandReceiver(port=port, default_height=0.2, timeout=0.1)
    tx = CommandSender("127.0.0.1", port)

    def close(a, *b):
        return all(math.isclose(x, y, abs_tol=1e-6) for x, y in zip(a, b, strict=True))

    def step(sleep=0.01):
        time.sleep(sleep)
        return rx.poll()

    a = rx.poll()
    assert a.timed_out and a.vx == 0.0 and a.height == 0.2, a

    tx.send(0.1, 0.05, 0.2, 0.22)
    a = step()
    assert not a.timed_out and close((a.vx, a.vy, a.wz, a.height), 0.1, 0.05, 0.2, 0.22), a

    tx.send(0.2, 0.1, 0.5, 0.24)  # 경계값은 f32 오차가 있어도 잘림이 아니다
    a = step()
    assert not a.clamped, a

    tx.send(1.0, 0.0, -9.0, 0.5)  # 범위 밖 → 잘림
    a = step()
    assert a.clamped and close((a.vx, a.wz, a.height), 0.2, -0.5, 0.24), a

    old = Command(0, now_ms(), Mode.POLICY, 0, 0, 0.1, 0.0, 0.0, 0.2)  # 예전 seq → 버림
    tx.sock.sendto(old.pack(), tx.target)
    a = step()
    assert close((a.vx,), 0.2), a

    tx.sock.sendto(b"garbage", tx.target)  # 형식 틀림 → 버림
    nan = Command(tx.seq + 5, now_ms(), Mode.POLICY, 0, 0, math.nan, 0.0, 0.0, 0.2)
    tx.sock.sendto(nan.pack(), tx.target)  # NaN → 버림
    a = step()
    assert close((a.vx,), 0.2), a

    other = CommandSender("127.0.0.1", port)  # 다른 조종기 → 무시
    other.send(-0.1, 0.0, 0.0, 0.2)
    a = step()
    assert close((a.vx,), 0.2), a

    tx.send(0.1, 0.0, 0.0, 0.2, estop=True)  # 비상정지 → 속도 0
    a = step()
    assert a.estop and a.mode == Mode.STOP and a.vx == 0.0, a

    tx.request_reset()
    tx.send(0.1, 0.0, 0.0, 0.2)
    a = step()
    assert a.reset_requested and close((a.vx,), 0.1), a
    tx.send(0.1, 0.0, 0.0, 0.2)
    a = step()
    assert not a.reset_requested, a

    rx.send_state(meas=(0.09, 0.0, 0.01, 0.199))
    time.sleep(0.01)
    st = tx.poll_state()
    assert st is not None and st.echo_seq == tx.seq - 1 and close(st.meas, 0.09, 0.0, 0.01, 0.199), st
    assert tx.rtt_ms is not None and tx.rtt_ms < 100, tx.rtt_ms

    a = step(0.15)  # 타임아웃 → 속도 0, 높이 유지
    assert a.timed_out and a.vx == 0.0 and close((a.height,), 0.2), a

    other.send(-0.1, 0.0, 0.0, 0.18)  # 타임아웃 뒤에는 다른 조종기를 받아들인다 (seq 0부터)
    a = step()
    assert not a.timed_out and close((a.vx, a.height), -0.1, 0.18) and not a.reset_requested, a

    for s in (tx, other, rx):
        s.close()
    print("protocol selftest OK")


if __name__ == "__main__":
    _selftest()
