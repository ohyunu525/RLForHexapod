# scripts/set_joint_limits.py
# URDF 재변환 없이 USD의 관절 한계만 수정한다. (USD 단위: 도)
# robot.usda(루트 레이어)에 override로 저장되므로 payload 원본은 그대로 남는다.
# URDF의 <limit>도 같은 값으로 맞춰둘 것. 재변환하면 이 수정은 덮어써진다.
from pathlib import Path

from pxr import Usd, UsdPhysics

ROBOT_USDA = Path(__file__).resolve().parents[1] / "src" / "RLforHexapod" / "assets" / "robot" / "robot" / "robot.usda"

# 관절 이름 접미사 -> (lower, upper) [deg]
NEW_LIMITS = {
    "_femur": (0.0, 180.0),
    "_tibia": (-250.0, -70.0),
}

if __name__ == "__main__":
    stage = Usd.Stage.Open(str(ROBOT_USDA))
    changed = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        name = prim.GetName()
        for suffix, (lo, hi) in NEW_LIMITS.items():
            if name.endswith(suffix):
                joint = UsdPhysics.RevoluteJoint(prim)
                old = (joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get())
                joint.GetLowerLimitAttr().Set(lo)
                joint.GetUpperLimitAttr().Set(hi)
                print(f"{name}: {old[0]} ~ {old[1]}  ->  {lo} ~ {hi}")
                changed += 1

    if changed != 12:
        raise SystemExit(f"femur/tibia 관절 12개를 예상했지만 {changed}개를 찾음. 저장하지 않음.")
    stage.GetRootLayer().Save()
    print("저장:", ROBOT_USDA)
