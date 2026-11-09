현재 완전체 로봇의 변환·검증·평지 학습 방법은 [완전체 로봇 평지 학습](docs/full_robot_walk.md)을 참고하세요.

## 평지 속도·방향 전환 강화 학습

Isaac Lab이 설치된 Windows 머신에서 계단 학습 전의 평지 모델로 시작합니다.

```powershell
$python = (Resolve-Path .venv\Scripts\python.exe).Path
.\scripts\run_flat_commands.ps1 -PythonPath $python -Difficulty hard `
    -NumEnvs 5120 -MaxIterations 2000
```

기본 시작점은 `logs/rsl_rl/hexapod_walk_wsp_stop/2026-11-08_11-11-43_flat_wsp_stop_20261108T111057433Z_9016d164_0fa08112/model_1999.pt`입니다.
SHA-256 `725c9696e82ef25e4b13e20c648562d17be6c79a6c1bc6ebd81936fbf2590527`을 확인한 뒤 기존 보행 정책과 관측 정규화 통계를 그대로 가져옵니다.
변경된 명령 분포에 맞춰 critic·optimizer·탐색 표준편차는 새로 초기화하고 반복 번호 0부터 학습합니다. 관측 67개와 동작 18개는 유지합니다.
다른 평지 모델을 시작점으로 사용하려면 `-Checkpoint <model_*.pt>`를 지정합니다. 95개 관측을 사용하는 혼합 모델은 이 평지 학습에 호환되지 않습니다.

| `-Difficulty` | 전후진 속도 (m/s) | 좌우 속도 (m/s) | 회전 속도 (rad/s) | 명령 변경 간격 (초) |
|---|---|---|---|---|
| `standard` | −0.10 ~ 0.20 | −0.10 ~ 0.10 | −0.50 ~ 0.50 | 10 |
| `hard` (기본) | −0.15 ~ 0.30 | −0.15 ~ 0.15 | −0.75 ~ 0.75 | 3 ~ 5 |
| `expert` | −0.20 ~ 0.40 | −0.20 ~ 0.20 | −1.00 ~ 1.00 | 1.5 ~ 3 |
| `extreme` | −0.25 ~ 0.50 | −0.25 ~ 0.25 | −1.25 ~ 1.25 | 1 ~ 2 |

각 간격마다 속도와 회전 명령을 범위 안에서 다시 무작위로 뽑습니다. 정지 명령 확률은 기본 30%이며, 몸통 높이 명령과 20초 에피소드 길이는 기존 평지와 같습니다.
최고 난이도는 `-Difficulty extreme`으로 선택합니다.
전후·좌우·회전 속도 명령이 모두 0인 동안 여섯 발 중 하나라도 접지 기준을 충족하지 못하면 `stand_support_deficit` 벌점(보상 가중치 −1.0)을 적용합니다.
접지는 기존 WSP 기준에 따라 위쪽 접촉력이 1 N을 넘고 발끝이 바닥 높이 기준 −1~+3 cm 이내일 때 인정합니다.
개별 값을 지정하면 해당 항목만 프리셋 위에 덮어씁니다.

```powershell
.\scripts\run_flat_commands.ps1 -PythonPath $python -Difficulty hard `
    -LinVelX @(-0.15, 0.35) -LinVelY @(-0.15, 0.15) `
    -YawRate @(-0.9, 0.9) -CommandInterval @(2.0, 4.0) -StandingFraction 0.30
```

실행 계획만 확인하려면 `-DryRun`, 환경 생성과 정책 불러오기만 검증하려면 `-VerifyOnly -NumEnvs 64`를 붙입니다.
학습 결과와 실제 명령 설정은 별도 `logs/rsl_rl/hexapod_walk_commands` 폴더의 `initialization.json`, `training_status.json`, 환경 설정 파일에 기록합니다.
실행기는 출력과 오류를 `outputs/flat_commands_launch`에 저장하며, 진행 상황을 반복 조회하지 않습니다.
Python 직접 실행은 `scripts/train_flat_wsp.py --task Hexapod-Walk-Commands --command-difficulty hard`를 사용합니다.
개별 옵션은 `--lin-vel-x MIN MAX`, `--lin-vel-y MIN MAX`, `--yaw-rate MIN MAX`, `--command-interval MIN MAX`, `--standing-fraction`입니다.

## 혼합 평지·계단 학습

Isaac Lab이 설치된 Windows 머신에서 기존 혼합 PPO 체크포인트로 자동 커리큘럼을 시작합니다.

```powershell
.\scripts\run_mixed_terrain.ps1 -PythonPath $python -ResumeCheckpoint $mixedCheckpoint `
    -HeightCurriculum -NumEnvs 5120 -MaxIterations 2000
```

`$python`에는 Isaac Lab 환경의 `python.exe` 경로를, `$mixedCheckpoint`에는 이전 혼합 학습에서 저장한 `model_*.pt` 경로를 지정합니다.
actor, critic, 관측 정규화 통계, Adam 상태를 복원하고 다음 반복부터 `-MaxIterations`만큼 추가 학습합니다.
`submit_mixed_terrain.ps1`에서도 같은 옵션으로 작업을 제출할 수 있습니다.
실행 환경과 체크포인트 호환성만 확인하려면 `run_mixed_terrain.ps1`에 `-VerifyOnly`를 붙입니다.

- `-HeightCurriculum`을 켜면 기본 4cm에서 시작해 6 → 8 → 12 → 16cm 순으로 진행합니다. 현재 높이에서 계단 앞부터 출발하는 에피소드를 시작할 때 100개까지 평가 대상으로 지정합니다. 100개가 모두 끝나고 80개 이상이 성공하면 승급합니다. 진행 중인 에피소드는 기존 높이를 유지하고 다음 리셋부터 새 높이를 적용합니다. 이어 학습할 때는 같은 옵션을 지정하면 체크포인트에 저장된 커리큘럼 진행 상태도 복원합니다.
- `-RandomStairHeight`를 켜면 계단 리셋마다 단 높이를 −16cm에서 +16cm 사이에서 균등하게 뽑습니다. 양수는 오르막, 음수는 내리막입니다. 두 옵션을 함께 켜면 각 계단 리셋에서 50% 확률로 랜덤 높이를 사용하고, 나머지는 현재 커리큘럼 높이를 사용합니다. 승급은 커리큘럼 높이에서의 결과로 판단합니다.
- 평지 비율은 기본 50%입니다. 높이 옵션을 사용할 때는 `-FlatFraction`을 조절해도 두 지형에 환경을 하나 이상씩 배정해야 합니다. 두 높이 옵션을 생략하면 기존 고정 계단 모드로 실행합니다.

혼합 계단에서는 앞발이 다음 단의 모서리 4cm 이내에 접근하고 중간·뒷발의 지지 위치가 유지되면 앞발 들기 단계로 전환합니다. 커리큘럼 평가 대상 이외의 고정 높이 학습 에피소드는 먼 출발점과 첫 발 들기 출발점을 섞어 사용합니다. 승급 평가는 항상 먼 출발점에서 시작하며, 랜덤 높이 에피소드도 먼 출발점을 사용합니다.

Python에서 직접 실행할 때는 `--resume-checkpoint`, `--height-curriculum`, `--random-stair-height`를 사용합니다. 높이 옵션을 끄려면 생략하거나 `--no-height-curriculum`, `--no-random-stair-height`를 지정합니다.

## 계단 단독 task 개발 계획

말씀하신 3개에 빠진 것들을 더해서, 의존 관계 순서대로 정리했습니다. **★**가 새로 추가한 항목입니다.

## A. 계단 teacher 만들기

**1. 계단 정보 관측 (특권 정보)**
- teacher actor와 critic에 계단까지 x 거리, 다음 단 높이, 발 밑 지형 높이 등을 넣습니다.
- ★ **체크포인트 호환 문제:** 관측이 늘면 신경망 입력 크기가 바뀌어서 **걷기 체크포인트를 그대로 불러올 수 없습니다.** 걷기 task에도 같은 관측 항목을 넣고(평지면 0 같은 값) 다시 학습하든지, 계단 teacher는 처음부터 학습하든지 정해야 합니다.

**2. 진행 플래그**
- 앞다리가 1단에 닿음, 몸통이 1단 높이 이상, 뒷다리까지 1단 위, 꼭대기 도달 같은 플래그를 만듭니다.
- 한 번 켜지면 에피소드 동안 유지되고(latch), 리셋 때 꺼집니다.
- 이 플래그도 관측에 넣습니다. 정책이 지금 어느 단계인지 알 수 있게 하기 위해서입니다.

**3. 플래그 기반 보상**
- 플래그가 처음 켜질 때 한 번 주는 보너스(milestone)와, 꼭대기에 도달했을 때의 성공 보너스를 만듭니다.

**4. ★ 리셋 시작점 커리큘럼** (SO101의 핵심)
- 계단 앞, 앞다리만 1단 위, 몸 전체 1단 위, 2단 앞 등 단계별 시작 자세를 만들어 무작위로 시작시킵니다.
- 시작 높이를 그 위치의 계단 높이에 맞춰야 로봇이 계단에 박히거나 공중에서 떨어지지 않습니다.
- 시작 단계에 맞는 플래그도 함께 켜서 시작해야 합니다.

**5. ★ 성공 종료와 성공률 지표**
- 꼭대기에 도달하면 성공으로 에피소드를 종료합니다.
- `success_rate`를 로그에 남기고, 평가는 항상 0단계(계단 앞)에서 시작합니다.

**6. ★ 계단 높이 커리큘럼**
- 지금은 160mm 하나로 고정입니다. 40 → 80 → 120 → 160 → 300mm 순으로 올라가야 합니다.
- **300mm 계단 파일은 아직 없습니다.** 다리 길이가 femur 약 0.10m + tibia 약 0.15m라서, 300mm가 물리적으로 가능한지 먼저 확인이 필요합니다.

**7. ★ 계단 task 설정 정리**
- **명령:** heading을 정면 `(0, 0)`으로 고정하고, 정지 명령이 나오지 않게 `lin_vel_x` 하한을 0.1 이상으로 올립니다.
- **종료 페널티:** `is_terminated`를 추가합니다. 계단 폭 이탈 종료(`off_stairs`)는 이미 추가했습니다.
- **로그 폴더:** 계단 task가 걷기와 같은 `experiment_name="hexapod_walk"`를 써서 로그가 섞입니다. 계단용 runner 설정을 따로 만들어야 합니다.

## B. 기존 코드 정리

**8. ★ 이미 발견한 문제들**
- `ang_vel_z=(0, 0)`이라 heading 보정이 동작하지 않습니다.
- `base_height_progress`가 스폰 높이를 기준으로 시작합니다.
- config 기본 `num_envs=4`입니다. 실행 때 지정하지 않으면 데이터가 너무 적습니다.
- PPO 설정에 튜토리얼에서 남은 `HexapodCameraPPORunnerCfg`(`vial`, `wrist_rgb`)가 있습니다.

## C. student 증류

**9. ★ student 센서 결정 (결정이 필요합니다)**
- 실제 로봇에 무엇을 달 건지 정해야 합니다. 관절 센서와 IMU만 쓸지, 깊이 카메라나 일반 카메라를 달지에 따라 student 관측과 렌더러 설정이 완전히 달라집니다.

**10. ★ 증류 task와 설정**
- SO101의 `...-Camera` task처럼 student용 env 설정, `teacher_state` 관측 그룹, 증류 runner 설정, task 등록을 만듭니다.

## D. 실제 로봇 적용 (나중에)

**11. ★ 도메인 랜덤화**
- 마찰, 질량, 모터 게인, 관측 노이즈, 외부에서 밀기 등을 무작위로 바꿉니다. MG996R 서보 특성도 맞춰야 합니다.

**12. ★ 검증 절차**
- 학습 전에 `zero_agent` 스모크 테스트를 돌려 봅니다.
- 평가 스크립트로 0단계 시작 성공률을 측정합니다.

---

**바로 시작할 수 있는 순서:** 8 → 7 → 1 → 2·3 → 5 → 4 → 6 → (9 결정) → 10 → 11

9번 student 센서는 미리 알려주시면, 1번 관측을 설계할 때 student가 나중에 따라 할 수 있는 형태로 맞출 수 있습니다. 8번부터 바로 진행할까요?
