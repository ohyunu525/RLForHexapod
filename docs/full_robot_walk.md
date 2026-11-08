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

## 평지 WSP: 접힌 무릎 출력의 포화 완화

**현재 결과: 정상 보행 수정은 미완료다.** 원격 RTX 3070에서 WSP 후보 12개를 평가했지만
보행 형태와 직진 성능을 함께 만족하지 못해 모두 미채택했다. 원본을 교체하지 않았다.
아래 도구는 재현 가능한 실험용이며, 검증된 정상 보행 체크포인트를 생성한다는 의미가 아니다.

2026-11-08에 받은 최신 평지 기준은
`logs/rsl_rl/hexapod_walk/2026-11-08_10-24-48_full_robot_resume_20261108_102439/model_13250.pt`다.
SHA-256은 `631f6af6f01b5b836c8c83fa7a25d74ce985a112226d217a1938f1c277f38493`이며,
기존 계단용 GRU 체크포인트와 다른 67관측·18행동 MLP 정책이다.

뒷발 하나만 움직이고 나머지 발은 접힌 채 모인다는 관찰을 바탕으로 출력 포화를 점검했다.
구성한 초기 자세 관측에 전진 0.2m/s 명령을 넣으면 L1~L5 tibia 출력은 모두 -1로 잘리고
L6 tibia는 약 +0.870이다. 이것은 실제 보행 궤적이 아닌 신경망 입력 검사다.
누적 정규화 통계에도 다리 비대칭이 있지만 현재 보행의 접촉·위상을 대신하지 않는다.
재구성한 actor는 export된 TorchScript와 같은 가중치에서 출력이 일치한다.
정규화는 `(obs - mean) / (std + 0.01)`이며, 작은 epsilon을 임의로 쓰면 다른 정책이 된다.

`scripts/flat_wsp.py`는 전진 상태의 과도한 음수 tibia 출력을 부드럽게 내부 범위로 당기는
목표를 만들고, 기본 설정에서는 마지막 출력층의 tibia 여섯 행만 ridge 선형해로 수정한다. 정지·회전과
포화되지 않은 출력에는 보호 가중치를 둔다. 보정량은 기록된 입력에서 보호 상태의
적용 행동 변화 0.1, 전체 변화 `strength` 이내로 되돌려 제한한다. 이 제한은 관측한
입력에만 해당하며 실제 폐루프 보행 안정성을 보장하지 않는다. 다른 다리를 L6와 같은
위상으로 복사하지 않고, 원본·정규화·상위 actor·coxa/femur·critic은 유지한다.
optimizer는 호환성을 위해 보존하므로 수정 후보로 PPO를 바로 resume하는 용도는 아니다.

명시적으로 `--joint-scope all`을 지정하면 coxa/femur의 양·음 포화까지 보정한다.
`--protected-cap`, `--max-applied-change`로 실험 범위를 넓힐 수 있으며 실제 적용한
제한과 측정된 변화는 JSON에 기록된다. 범위를 넓혔다고 보행이 개선됐다는 뜻은 아니다.

학습 PC에서 원본 기록 → 실제 관측으로 WSP 후보 생성 → 동일 조건 비교를 실행한다.

```powershell
uv run --locked python scripts/run_flat_wsp.py --dry-run
uv run --locked python scripts/run_flat_wsp.py
```

기본은 환경 4개, 조건당 750 step(15초), seed 73이다. 전진 0.1/0.15/0.2m/s,
정지, 좌·우 회전, 낮은/높은 몸통 높이를 순서대로 평가한다. 각 실행은 새
`outputs/flat_wsp/compare_...` 디렉터리를 만든다. 복제된 환경 개수나 seed 개수를
독립 시행 횟수로 해석하지 않는다. 체크포인트를 바꾸면 생성기의 해시 고정부터 검토해야 한다.

평가기는 자동 리셋 전 상태와 실패 프레임을 저장한다. 발별 무릎의 양방향 이동량·왕복 횟수,
관절 범위·목표 포화, tibia 접촉, 속도·추종 오차·몸통 운동·실패 횟수를 기록한다.
발끝 이동량은 몸통 좌표 기준이며, 리셋에 의한 순간이동을 보폭으로 세지 않는다.
tibia 접촉은 링크 전체의 접촉이므로 발끝 지지를 확정하는 지표가 아니다.

`comparison.json`은 같은 자산·초기 상태·명령·평가 길이인지 확인하고 원본 대비 변화를
정리한다. 속도 저하·추종 오차 증가·실패 증가를 표시하지만 정책을 자동으로 채택하지 않는다.
원래 움직이지 않던 발들이 실제로 지지와 스윙을 반복하는지 보행 영상·접촉 기록도 확인해야 한다.
합성 입력만으로 후보를 만드는 `--synthetic-probes` 옵션은 진단용이며 물리 검증을 대신하지 않는다.

### 2026-11-08 실제 비교 결과

원본의 0.15m/s 명령에서 실제 평균 전진 속도는 0.1446m/s다. 초기 5초를 제외한 10초 동안
L3 tibia 접촉은 약 4.1%, L6 접촉은 0%였다. L6는 움직여도 지지하지 않으며, L3 무릎은
음수 출력 포화에 오래 걸려 있다. 단순한 모습 차이만으로 판단한 결과가 아니다.

| 보정 방식 | 0.15m/s 명령의 실제 전진 속도 | 판단 |
|---|---:|---|
| 무릎/전체 관절의 포화 완화 4개 | 0.1443~0.1456m/s | 지지하지 않는 뒷발 문제가 남음 |
| 지지 자세 중심 출력층 3개 | 0.0133~0.0430m/s | 발 배치가 바뀌지만 전진 성능 저하 |
| 교대 tripod 출력층 2개 | 0.0667~0.1011m/s | 무릎 움직임과 접촉이 늘지만 튀는 구간·추종 오차 증가 |
| 1.8Hz tripod 1개 | 0.0431m/s | 전진 성능 저하, 공중 구간이 남음 |
| 좌우 대칭 보정 2개 | -0.0141~0.0229m/s | 전진 성능 저하; 강한 보정은 후진 |

평가는 각 조건 4개 복제 환경·15초이며 실패 프레임도 포함한다. 대표 0.15m/s 비교에서는
낙상 종료가 없었지만, 그것만으로 정상 보행이라고 판정하지 않았다. 이 실험은 출력층 WSP
범위에서의 결과이며 모든 가중치 수정 방법이 불가능하다는 증명은 아니다.

상세 결과·체크포인트 해시·원시 궤적 위치는
`outputs/flat_wsp/physical_20261108/experiment_assessment.json`과 `decision_20261108.json`에 있다.
자세/위상/대칭 후보의 재현 스크립트도 같은 폴더에 보존했다. 최종 코드의 CPU 회귀검사
23개와 실제 4환경 smoke를 통과했으며, **정상 보행의 수용 기준은 통과하지 못했다.**

## WSP에서 새 보상으로 재학습

`Hexapod-Walk-WSP`는 기존 `Hexapod-Walk`와 분리된 평지 재학습 task다.
관측 67개·행동 18개와 로봇·명령 범위는 유지하고, 여섯 다리의 지지 참여와 뜀 억제를 추가한다.
각 다리의 지지율을 시정수 0.6초로 추적해 이동 중 20% 미만 또는 85% 초과인 다리에
벌점을 준다. 지지하지 않는 발과 계속 붙어 있는 발을 함께 다루며, 정지 명령에서는
이 교대 조건을 끈다. 특정 tripod 위상이나 발 순서를 강제하지 않는다.
지지는 tibia의 위쪽 힘이 1N을 넘고 발끝 높이가 지면 기준 -1~3cm일 때로 근사한다.
정확한 접촉점 판정은 아니므로 영상과 발끝 궤적 확인도 필요하다.
새 벌점 가중치는 다리별 지지 참여 -0.6, 모든 발이 뜬 상태 -0.8,
지지 발 3개 미만 -0.25이며 기존 수직속도 제곱 벌점은 -2에서 -4로 높인다.

출발 모델은 가장 뚜렷하게 다리 움직임이 달라진 `model_13250_wsp_tripod_compact_v2.pt`다.
이 모델은 0.15m/s 명령에서 평균 전진 0.1011m/s였고 몸이 뜨는 구간이 증가한 **실험용 초기값**이다.
재학습 전부터 정상 보행을 통과한 모델로 해석하지 않는다.

학습 PC의 저장소 루트에서 실행한다. 체크포인트는 이미 해당 PC의 아래 기본 경로에 있다.
`outputs/`는 Git에 포함되지 않으므로 다른 PC로 옮길 때는 모델 파일도 별도로 전달한다.

```powershell
powershell -NoProfile -NoExit -ExecutionPolicy Bypass -File .\scripts\train_flat_wsp.ps1 -MaxIterations 3000 -NumEnvs 1024
```

기본 모델 경로는
`outputs/flat_wsp/physical_20261108/tripod_probes/model_13250_wsp_tripod_compact_v2.pt`,
SHA-256은 `b002783a2c6ab42c5e92c27022c78ae6bccd79f5d12b9c3a3fb70079117b3059`다.
다른 초기 모델을 쓰려면 `-Checkpoint '모델 경로'`를 지정한다.
위 명령의 실행 정책 옵션은 해당 PowerShell 프로세스에만 적용한다.
`-NoExit`는 스크립트가 끝나거나 오류가 나도 PowerShell 창을 남겨 오류 내용을 확인할 수 있게 한다.
자동화·SSH에서 종료 코드를 받을 때는 `-NoExit`를 제외한다.

이 명령은 자산 검사와 중립 자세 검사를 통과한 뒤 새 학습을 시작한다.
actor MLP와 관측 정규화 통계만 불러오며, 탐색 표준편차는 0.10으로 설정한다.
critic·critic 정규화·Adam 상태·iteration은 새로 시작한다. 이전 보상으로 학습된 critic이나
13,250회차 optimizer를 재개하지 않는다. actor 정규화의 평균·표준편차·count와 epsilon 0.01은 보존한다.
새 실행은 `logs/rsl_rl/hexapod_walk_wsp/`에 저장된다.
`initialization.json`에는 가중치 일치·초기화·모델 해시 확인 결과,
`training_status.json`에는 실행 결과가 남는다. TensorBoard의 `Metrics/WSP/`에는
L1~L6 지지율, 이동 명령 중 지지율, 모든 발이 뜬 비율, 지지 발 수, 수직속도 RMS를 기록한다.

실행기의 단계별 표준 출력과 오류는 `outputs/flat_wsp_launch/<실행 이름>/`에 저장한다.
`launcher_status.json`에서 진행 단계·프로세스 PID·종료 코드를 확인하고,
학습 출력은 `03_train.stdout.log`, 오류는 `03_train.stderr.log`에서 확인한다.
학습 프로세스는 파일로 출력을 보내고 화면은 해당 로그를 읽어 표시하므로,
화면 출력 연결 오류가 학습의 표준 출력으로 전파되는 경로를 분리했다.
Python에서 예외가 발생하면 `training_status.json`에도 전체 traceback과 마지막 저장 모델 목록을 남긴다.
Ctrl+C는 실행기가 시작한 작업 프로세스 트리를 중지한다.

기본 PPO 학습률은 고정 `1e-4`, clip 0.15, entropy 0.002다. 이는 조정을 시작할 값이며
3,000회 학습으로 보행 개선이 보장된다는 뜻은 아니다. 학습 완료 후 같은 명령·자산·시간으로
원본과 비교해 L3/L6 지지, 발별 스윙, 전진 속도, 공중 구간을 다시 확인한다.

짧은 실행 검사 또는 명령 확인은 다음과 같다.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_flat_wsp.ps1 -MaxIterations 2 -NumEnvs 64
powershell -ExecutionPolicy Bypass -File .\scripts\train_flat_wsp.ps1 -DryRun
```

2026-11-08 학습 PC RTX 3070에서 자산 검사, 4환경 중립 자세 1,100 step 검사,
**새 task 1,024환경·2회 PPO 업데이트와 체크포인트 저장을 통과했다.**
새 보상·warm-start CPU 검사 11개도 같은 PC에서 통과했다. 로컬 WSP 관련 회귀검사는 총 34개다.
실행 기록은
`logs/rsl_rl/hexapod_walk_wsp/2026-11-08_02-59-11_flat_wsp_20261108T025848548Z_638e0aee_6105dc3b/`에 있다.
이 짧은 검사는 초기화·실행·로그·저장 검증이며, **재학습 후 정상 보행이나 속도 개선 검증은 아니다.**

이후 5,012환경·2,000회 설정으로 시작한 실행 두 개가 `WinError 233`으로 끝났다.
마지막 실행은 TensorBoard에 87회차까지 기록되고 `model_50.pt`가 저장되어 있었다.
검사한 가중치와 손실은 유한했고 기록된 NaN·낙상 종료는 없었다. 오류는 출력 파이프가
끊긴 상황과 일치하지만 당시 traceback이 없어 정확한 호출 위치까지 확정하지는 못했다.
위의 파일 출력·실행 상태 기록과 `-NoExit` 안내는 이 실패를 진단하고 재발 경로를 줄이기 위한 변경이다.

수정 후 Windows PowerShell 5.1에서 실행기 검사 3개를 통과했다. 인수의 공백·따옴표 보존,
실패 종료 코드와 후속 단계 중단, 부모 출력 파이프를 닫아도 작업과 파일 로그가 완료되는 경우를 검사했다.
학습 PC에서는 같은 5,012환경 설정으로 **5회 학습·저장을 완료**했고 세 단계 종료 코드는 모두 0이었다.
실행기 로그는 `outputs/flat_wsp_launch/flat_wsp_20261108T050553006Z_3227f3f2/`,
학습 로그는
`logs/rsl_rl/hexapod_walk_wsp/2026-11-08_05-06-19_flat_wsp_20261108T050553006Z_3227f3f2_1b4725e0/`다.
2,000회 전체 학습의 완료를 검증한 것은 아니다.
