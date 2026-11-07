"""Onshape export를 학습용 URDF로 정리한다. Onshape API는 쓰지 않는다.

입력: cad/onshape_export/ (onshape-to-robot 결과, git 제외)
출력: src/RLforHexapod/assets/urdf/robot.urdf, src/RLforHexapod/assets/urdf/meshes/

하는 일 (몸통 링크만 바꾸고, 다리 링크는 메시 경로만 바꾼다)
0. 몸통 좌표계를 z축으로 180° 돌려 라이다 쪽을 +x(앞)로 만든다. CAD에서는 라이다가 -x 쪽에 있다.
   돌리고 나면 L4가 예전 L1 자리에 같은 자세로 온다 (L4↔L1, L5↔L2, L6↔L3).
1. 로봇 밖에 떨어진 부품을 뺀다. CAD에서 mate가 빠진 XT60 커넥터가 몸통 옆 36cm에 있었다.
2. Onshape 질량이 0이거나 PLA 밀도로 계산된 부품을 실측 질량으로 바꾸고,
   몸통의 질량, 무게중심, 관성을 다시 계산한다. 질량은 부품 메시 부피에 고르게 나눈다.
3. 작은 부품 약 780개(라즈베리파이, PCB, 라이다 내부)를 묶음마다 볼록 껍질 하나로 바꾼다.
   MJWarp는 메시마다 볼록 껍질 충돌체를 만들기 때문에, 그대로 두면 몸통 충돌체가 798개가 된다.

    python scripts/postprocess_urdf.py

필요한 패키지: numpy, numpy-stl, scipy
"""

import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation
from stl import Mode
from stl import mesh as stl_mesh

REPO = Path(__file__).resolve().parents[1]
EXPORT = REPO / "cad" / "onshape_export"
OUT = REPO / "src" / "RLforHexapod" / "assets" / "urdf"
MESH_DIR = "meshes"

# 질량을 바꿀 몸통 부품 묶음. 키는 Onshape part studio의 elementId (.part 파일에 있음).
#   onshape: export에 들어간 질량 [kg], measured: 실측 질량 [kg] (2026-11-07)
#   hull: True면 묶음 전체를 볼록 껍질 하나로 바꾼다
BODY_GROUPS = {
    "lidar": {  # Unitree L2 본체
        "elements": ["71191938159602e72ea12258"], "onshape": 0.0, "measured": 0.247, "hull": True},
    "rpi5": {  # 라즈베리파이 5 + 방열판·팬
        "elements": ["f786ae28b0ab15a1a8ec226a", "593e4b6b7cc9ecd514934104"],
        "onshape": 0.0, "measured": 0.068, "hull": True},
    "pcb": {  # 서보 보드. 레귤레이터 2개(DSN6000AUD, D24V90F5)는 Onshape 입력값을 그대로 쓴다
        "elements": ["b015a5e5c29395bf8fcc5916"], "onshape": 0.0, "measured": 0.063, "hull": True},
    "lidar_cap": {
        "elements": ["5215621263a74a5c40777cab"], "onshape": 0.072093, "measured": 0.041, "hull": False},
    "cap": {
        "elements": ["e95ffc0e93c265f1a845add3"], "onshape": 0.051058, "measured": 0.03098, "hull": False},
    "body_top2": {  # bodyTop <2>. 같은 이름의 원래 bodyTop(10.25g)은 Onshape 입력값을 그대로 쓴다
        "elements": ["4f6fc3828b56ea281fbd70d7"], "onshape": 0.012235, "measured": 0.01035, "hull": False},
}

# 몸통 원점에서 이 거리[m]보다 바깥(x 또는 y)에 점이 있는 부품은 떨어진 부품으로 보고 뺀다. 몸통 반폭은 0.117m
STRAY_LIMIT = 0.2
# 몸통 좌표계 회전 [rad]. 라이다가 달린 쪽이 로봇의 앞(+x)이다
BASE_YAW = np.pi
HULL_RGBA = "0.3 0.3 0.3 1"
# 볼록 껍질 묶음의 겉모습에서 이보다 작은 부품(핀, 칩 다리 등, 외곽 대각선 [m])은 뺀다. 질량과 충돌에는 영향 없음
MIN_VISUAL_SIZE = 0.005


def parse_vec(text: str) -> np.ndarray:
    return np.array([float(v) for v in text.split()])


def stem_of(filename: str) -> str:
    return Path(filename.replace("package://", "")).stem


def part_element(stem: str) -> str:
    text = (EXPORT / "assets" / f"{stem}.part").read_text(encoding="utf8")
    return re.search(r'"elementId":\s*"(\w+)"', text).group(1)


def load_in_link_frame(geom_elem: ET.Element) -> np.ndarray:
    """visual/collision 요소의 메시를 링크 좌표계 삼각형 (T, 3, 3)으로 읽는다."""
    origin = geom_elem.find("origin")
    stem = stem_of(geom_elem.find("geometry/mesh").get("filename"))
    tris = stl_mesh.Mesh.from_file(str(EXPORT / "assets" / f"{stem}.stl")).vectors.astype(np.float64)
    rot = Rotation.from_euler("xyz", parse_vec(origin.get("rpy")))
    return rot.apply(tris.reshape(-1, 3)).reshape(tris.shape) + parse_vec(origin.get("xyz"))


def mass_props(tris: np.ndarray):
    """닫힌 메시의 부피, 부피중심, 부피중심 기준 관성(밀도 1)."""
    m = stl_mesh.Mesh(np.zeros(len(tris), dtype=stl_mesh.Mesh.dtype))
    m.vectors = tris
    volume, cog, inertia = m.get_mass_properties()
    if volume < 0:  # 법선이 안쪽을 향한 메시
        volume, inertia = -volume, -inertia
    return volume, cog, inertia


def combine(bodies):
    """(질량, 무게중심, 무게중심 기준 관성) 목록을 하나로 합친다. 음수 질량은 빼기."""
    mass = sum(m for m, _, _ in bodies)
    com = sum(m * c for m, c, _ in bodies) / mass
    inertia = np.zeros((3, 3))
    for m, c, i in bodies:
        d = c - com
        inertia += i + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return mass, com, inertia


def is_closed(tris: np.ndarray) -> bool:
    """모든 모서리를 정확히 두 삼각형이 공유하면 닫힌 메시."""
    _, ids = np.unique(tris.reshape(-1, 3).round(9), axis=0, return_inverse=True)  # 꼭짓점 번호
    ids = ids.reshape(-1, 3)
    edges = np.sort(np.concatenate([ids[:, [0, 1]], ids[:, [1, 2]], ids[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool((counts == 2).all())


def distribute(tris_list, mass):
    """질량을 메시 부피에 고르게 나눴을 때의 (질량, 무게중심, 관성). 닫히지 않은 메시는 부피를 잴 수 없어 뺀다."""
    props = [mass_props(t) for t in tris_list if is_closed(t)]
    volume = sum(v for v, _, _ in props)
    density = mass / volume
    return combine([(density * v, c, density * i) for v, c, i in props])


def hull_triangles(points: np.ndarray) -> np.ndarray:
    """바깥을 향하는 볼록 껍질 삼각형 (T, 3, 3)."""
    try:
        hull = ConvexHull(points)
    except QhullError:
        hull = ConvexHull(points, qhull_options="QJ")  # 납작한 부품
    tris = points[hull.simplices]
    center = points.mean(axis=0)
    normal = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    inward = np.einsum("ij,ij->i", normal, tris.mean(axis=1) - center) < 0
    tris[inward] = tris[inward][:, [0, 2, 1]]
    return tris


def save_stl(tris: np.ndarray, path: Path):
    m = stl_mesh.Mesh(np.zeros(len(tris), dtype=stl_mesh.Mesh.dtype))
    m.vectors = tris
    m.get_header = lambda name: "postprocess_urdf".ljust(80)  # 시간 기록을 빼서 결과가 매번 같게
    m.save(str(path), mode=Mode.BINARY)


def mesh_elem(tag: str, filename: str, rgba: str | None = None) -> ET.Element:
    elem = ET.Element(tag)
    ET.SubElement(elem, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(ET.SubElement(elem, "geometry"), "mesh", filename=filename)
    if rgba is not None:
        ET.SubElement(ET.SubElement(elem, "material", name=f"{Path(filename).stem}_material"), "color", rgba=rgba)
    return elem


def set_inertial(link: ET.Element, mass: float, com: np.ndarray, inertia: np.ndarray):
    inertial = link.find("inertial")
    inertial.find("origin").set("xyz", " ".join(f"{v:.6g}" for v in com))
    inertial.find("origin").set("rpy", "0 0 0")
    inertial.find("mass").set("value", f"{mass:.6g}")
    names = {"ixx": (0, 0), "ixy": (0, 1), "ixz": (0, 2), "iyy": (1, 1), "iyz": (1, 2), "izz": (2, 2)}
    for key, (r, c) in names.items():
        inertial.find("inertia").set(key, f"{inertia[r, c]:.6g}")


def link_inertial(link: ET.Element):
    inertial = link.find("inertial")
    i = inertial.find("inertia")
    g = lambda k: float(i.get(k))  # noqa: E731
    inertia = np.array([[g("ixx"), g("ixy"), g("ixz")], [g("ixy"), g("iyy"), g("iyz")], [g("ixz"), g("iyz"), g("izz")]])
    rpy = parse_vec(inertial.find("origin").get("rpy"))
    if np.abs(rpy).max() > 1e-9:
        raise SystemExit("몸통 관성 origin에 회전이 있습니다. 이 스크립트는 회전 없는 경우만 처리합니다.")
    return float(inertial.find("mass").get("value")), parse_vec(inertial.find("origin").get("xyz")), inertia


def part_key(elem: ET.Element):
    """부품 하나를 가리키는 키. 같은 STL을 여러 부품이 함께 쓰므로 위치까지 넣는다."""
    origin = elem.find("origin")
    return elem.find("geometry/mesh").get("filename"), origin.get("xyz"), origin.get("rpy")


def process_body(body: ET.Element, mesh_out: Path) -> set[str]:
    """몸통 링크를 고치고, 그대로 쓰는 메시 이름들을 돌려준다."""
    element_to_group = {e: name for name, g in BODY_GROUPS.items() for e in g["elements"]}
    group_tris = {name: [] for name in BODY_GROUPS}
    kept, stray = set(), []

    for visual in body.findall("visual"):
        stem = stem_of(visual.find("geometry/mesh").get("filename"))
        tris = load_in_link_frame(visual)
        if np.abs(tris[..., :2]).max() > STRAY_LIMIT:
            stray.append(stem)
            continue
        group = element_to_group.get(part_element(stem))
        if group is not None:
            group_tris[group].append(tris)
        if group is None or not BODY_GROUPS[group]["hull"]:
            kept.add(part_key(visual))
    for name, g in BODY_GROUPS.items():
        if not group_tris[name]:
            raise SystemExit(f"묶음 {name}의 부품을 찾지 못했습니다. elementId를 확인하세요.")

    # 질량: export 값에서 묶음마다 (실측 - Onshape) 만큼을 그 묶음의 메시 부피에 고르게 더한다
    mass0, com0, inertia0 = link_inertial(body)
    parts = [(mass0, com0, inertia0)]
    for name, g in BODY_GROUPS.items():
        parts.append(distribute(group_tris[name], g["measured"] - g["onshape"]))
    mass, com, inertia = combine(parts)
    if np.linalg.eigvalsh(inertia).min() <= 0:
        raise SystemExit("계산한 관성이 양의 정부호가 아닙니다.")
    set_inertial(body, mass, com, inertia)

    # 메시: 그대로 쓰는 부품은 경로만 바꾸고, 볼록 껍질 묶음은 새 메시 하나로 바꾼다
    kept_stems = set()
    for tag in ("visual", "collision"):
        for elem in body.findall(tag):
            if part_key(elem) in kept:
                stem = stem_of(elem.find("geometry/mesh").get("filename"))
                elem.find("geometry/mesh").set("filename", f"{MESH_DIR}/{stem}.stl")
                kept_stems.add(stem)
            else:
                body.remove(elem)
    for name, g in BODY_GROUPS.items():
        if not g["hull"]:
            continue
        # 겉모습: 부품마다 볼록 껍질, 충돌: 묶음 전체의 볼록 껍질 하나
        points = [t.reshape(-1, 3) for t in group_tris[name]]
        visual = np.concatenate([hull_triangles(p) for p in points
                                 if np.linalg.norm(p.max(0) - p.min(0)) >= MIN_VISUAL_SIZE])
        collision = hull_triangles(np.concatenate([t.reshape(-1, 3) for t in group_tris[name]]))
        save_stl(visual, mesh_out / f"body_{name}.stl")
        save_stl(collision, mesh_out / f"body_{name}_hull.stl")
        body.append(mesh_elem("visual", f"{MESH_DIR}/body_{name}.stl", HULL_RGBA))
        body.append(mesh_elem("collision", f"{MESH_DIR}/body_{name}_hull.stl"))

    print(f"떨어진 부품 제거: {stray}")
    print(f"몸통 질량 {mass0 * 1000:.1f} g -> {mass * 1000:.1f} g")
    print(f"몸통 무게중심 {np.round(com0 * 1000, 1)} mm -> {np.round(com * 1000, 1)} mm")
    print(f"몸통 메시 visual {len(body.findall('visual'))}개, collision {len(body.findall('collision'))}개")
    return kept_stems


def fmt(vec) -> str:
    return " ".join(f"{v:.9g}" for v in np.where(np.abs(vec) < 1e-12, 0.0, vec))


def rotate_body_frame(robot: ET.Element, yaw: float):
    """몸통 링크 좌표계를 z축으로 yaw만큼 돌린다. 몸통에 붙은 것(관성, 메시, 다리 관절)의 위치를 새 좌표계로 옮긴다."""
    rot = Rotation.from_euler("z", -yaw)  # 예전 좌표 -> 새 좌표
    body = robot.find("link[@name='body']")

    def move(origin: ET.Element):
        origin.set("xyz", fmt(rot.apply(parse_vec(origin.get("xyz")))))
        new = rot * Rotation.from_euler("xyz", parse_vec(origin.get("rpy")))
        origin.set("rpy", fmt(new.as_euler("xyz")))

    for elem in body.findall("visual") + body.findall("collision"):
        move(elem.find("origin"))
    for joint in robot.findall("joint"):
        if joint.find("parent").get("link") == "body":
            move(joint.find("origin"))

    mass, com, inertia = link_inertial(body)
    r = rot.as_matrix()
    set_inertial(body, mass, rot.apply(com), r @ inertia @ r.T)


def main():
    tree = ET.parse(EXPORT / "robot.urdf")
    robot = tree.getroot()
    mesh_out = OUT / MESH_DIR
    shutil.rmtree(mesh_out, ignore_errors=True)
    mesh_out.mkdir(parents=True)

    rotate_body_frame(robot, BASE_YAW)
    used = process_body(robot.find("link[@name='body']"), mesh_out)
    for link in robot.findall("link"):
        if link.get("name") == "body":
            continue
        for mesh in link.iter("mesh"):
            stem = stem_of(mesh.get("filename"))
            mesh.set("filename", f"{MESH_DIR}/{stem}.stl")
            used.add(stem)
    for stem in sorted(used):
        shutil.copyfile(EXPORT / "assets" / f"{stem}.stl", mesh_out / f"{stem}.stl")

    total = sum(float(m.get("value")) for m in robot.iter("mass"))
    print(f"로봇 전체 질량 {total * 1000:.1f} g")

    ET.indent(tree, space="  ")
    tree.write(OUT / "robot.urdf", encoding="utf-8", xml_declaration=True)
    print("저장:", OUT / "robot.urdf")


if __name__ == "__main__":
    main()
