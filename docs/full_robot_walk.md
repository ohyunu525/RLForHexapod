# 완전체 로봇 평지 학습

2026-11-07에 갱신된 `assets/urdf/robot.urdf`를 변환한 자산을 사용한다.
라이다·라즈베리파이 등이 포함된 형상과 질량을 반영하며, 라이다 관측이나 실물 센서 I/O를 추가하는 작업은 아니다.

- 실제 로딩 경로: `src/RLforHexapod/assets/robot/full/robot/robot.usda`
- 링크 19개, 회전 관절 18개, 총 질량 2.28614kg, 몸통 질량 1.28096kg.
- 물리 variant는 `physics`, 베이스는 자유롭게 움직인다. URDF의 충돌 형상을 사용한다.
- 새 URDF의 앞다리 L3/L4는 몸통 +x에 있다. Isaac Lab 3.0의 초기 quaternion은 **xyzw** `(0, 0, 0, 1)`이다.
  `(1, 0, 0, 0)`은 x축 180° 회전으로 로봇을 뒤집으므로 사용하면 안 된다.
- `Hexapod-Walk` 관측 67개, 행동 18개. 모터·보상·명령 범위는 현재 기준값을 유지한다.
- 공통 로봇 설정을 쓰는 계단 task에도 같은 자산이 적용된다. 이전 로봇 체크포인트의 평가와 직접 비교하지 않는다.

## 학습 PC에서 실행

이 변경이 포함된 코드를 받은 뒤 저장소 루트의 PowerShell에서 실행한다.
USD 파일은 Git LFS이므로 실제 파일을 받아야 한다.

```powershell
git lfs pull
.\scripts\train_full_robot.ps1
```

실행 순서는 자산 검사 → 환경 4개에서 중립 행동 1,100 step 검사 → 새 평지 teacher 학습이다.
검사 실패 시 다음 단계는 실행하지 않는다. 중립 자세 검사는 보행 성능 평가가 아니다.
학습 기본값은 환경 4,096개, seed 42, CUDA 0, 800회이며 기존 체크포인트를 재개하지 않는다.
실행 이름은 `full_robot_flat_날짜_고유번호`, 로그는 `logs/rsl_rl/hexapod_walk/`에 생성된다.
기존 학습 프로세스를 종료하거나 기존 보고서를 덮어쓰지 않는다.

반복 수·환경 수를 바꾸거나 실행할 명령만 확인할 수 있다.

```powershell
.\scripts\train_full_robot.ps1 -MaxIterations 3000 -NumEnvs 4096
.\scripts\train_full_robot.ps1 -DryRun
```

`outputs/*_smoke.json`에는 실제 자산 경로, 관절 순서, 초기 앞쪽·위쪽 방향, 투영 중력, 기울기, 높이 범위,
리셋 전 NaN 검사, 종료 원인이 기록된다. 첫 실패에서는 리셋 전 자세와 속도를 저장하고 즉시 중단한다.
초기 자세가 무너지면 이 보고서로 원인을 확인한다.
검증을 마치기 전 임의의 두 실험으로 분기하지 않는다. 서로 다른 옵션을 비교하게 되면
동일한 URDF/USD·관측·평가 조건을 유지하고 변경 옵션만 분리한다.

## USD 재생성

학습 PC는 이미 생성된 USD를 사용하므로 변환 도구 설치가 필요 없다.
URDF 또는 메시가 바뀐 경우에만 Python 3.12의 별도 환경에서 재생성한다.
변환 환경을 학습 환경과 분리해 `uv.lock`의 런타임 의존성을 유지한다.

```powershell
uv venv outputs/asset-converter --python 3.12
uv pip install --python outputs/asset-converter/Scripts/python.exe -r scripts/requirements-asset-conversion.txt
& outputs/asset-converter/Scripts/python.exe scripts/convert_urdf.py --force
& outputs/asset-converter/Scripts/python.exe scripts/validate_robot_asset.py --require-manifest
```

변환은 GPU나 Isaac Sim 화면 없이 실행된다. 임시 폴더에서 검증한 다음 고정 출력 경로로 옮긴다.
`conversion_manifest.json`에 URDF·메시 해시와 변환 옵션을 기록한다.
학습 실행 스크립트는 이 기록과 원본이 일치하는지도 검사해 오래된 USD로 학습하는 일을 막는다.
새 USD와 payload 전체, manifest를 함께 전달해야 한다.

## 현재 검증 범위

자산 변환과 XML/USD 검사, 검증기의 단위 테스트, 실행 스크립트의 구문·명령 구성을 로컬에서 확인했다.
11월 8일 원격 점검에서는 첫 step부터 `bad_ori`가 발생했다. 이전 코드가 초기 회전을
`wxyz`로 잘못 해석한 것이 확인돼 `xyzw` identity로 복구했다.
[고정된 Isaac Lab 초기 회전 정의](https://github.com/isaac-sim/IsaacLab/blob/270a8c1d4a26181adf2c5bd2fc9a3c649f17a294/source/isaaclab/isaaclab/assets/asset_base_cfg.py#L38-L40)를 기준으로,
공식 회전 함수를 CPU에서 실행해 이전 값의 기울기는 π, 수정 값은 0임을 확인했다.
로컬에는 프로젝트의 CUDA 실행 환경이 없으므로 수정 후 GPU 점검과 PPO 학습은 아직 검증되지 않았다.
원격에서 `./scripts/train_full_robot.ps1 -NumEnvs 64 -MaxIterations 2`를 실행해
smoke 통과, 학습 로그 갱신, 체크포인트 저장을 확인한 뒤 장시간 학습한다.
이 평지 teacher는 이후 계단 작업의 기반이지만, 계단 teacher의 관측 차원이 달라 그대로 resume할 수는 없다.
