import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from pathlib import Path

# ── YOLO Pose 초기화 (캐시) ────────────────────────────────────
@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO("yolo11n-pose.pt")  # 자동 다운로드 (~7MB)

# ── COCO 17 keypoint 정의 (몸통 관심 포인트 강조) ──────────────
KEYPOINTS = {
    0:  ("코 (Nose)",              (255, 200,   0), True),
    5:  ("왼쪽 어깨 (L Shoulder)", (  0, 120, 255), True),
    6:  ("오른쪽 어깨 (R Shoulder)",(  0,  80, 200), True),
    7:  ("왼쪽 팔꿈치 (L Elbow)",  (  0, 200,  80), True),
    8:  ("오른쪽 팔꿈치 (R Elbow)",(  0, 160,  60), True),
    9:  ("왼쪽 손목 (L Wrist)",    (255, 140,   0), True),
    10: ("오른쪽 손목 (R Wrist)",  (200, 100,   0), True),
    11: ("왼쪽 엉덩이 (L Hip)",    (200,   0,  80), True),
    12: ("오른쪽 엉덩이 (R Hip)",  (160,   0,  60), True),
    13: ("왼쪽 무릎 (L Knee)",     (140,   0, 200), True),
    14: ("오른쪽 무릎 (R Knee)",   (100,   0, 160), True),
}

CONNECTIONS = [
    (5, 6),   # 어깨
    (5, 7),   (7, 9),   # 왼팔
    (6, 8),   (8, 10),  # 오른팔
    (5, 11),  (6, 12),  # 상체
    (11, 12),           # 엉덩이
    (11, 13), (13, 15), # 왼다리 (무릎까지)
    (12, 14), (14, 16), # 오른다리 (무릎까지)
    (0, 5),   (0, 6),   # 코-어깨
]


def draw_pose(image_rgb: np.ndarray, keypoints_xy: np.ndarray, confs: np.ndarray) -> np.ndarray:
    img = image_rgb.copy()
    h, w = img.shape[:2]

    # 연결선
    for (a, b) in CONNECTIONS:
        if a < len(keypoints_xy) and b < len(keypoints_xy):
            ca = confs[a] if a < len(confs) else 0
            cb = confs[b] if b < len(confs) else 0
            if ca > 0.3 and cb > 0.3:
                x1, y1 = int(keypoints_xy[a][0]), int(keypoints_xy[a][1])
                x2, y2 = int(keypoints_xy[b][0]), int(keypoints_xy[b][1])
                cv2.line(img, (x1, y1), (x2, y2), (200, 200, 200), 2, cv2.LINE_AA)

    # 포인트
    for idx, (label, color, _) in KEYPOINTS.items():
        if idx >= len(keypoints_xy):
            continue
        conf = confs[idx] if idx < len(confs) else 0
        if conf > 0.3:
            cx, cy = int(keypoints_xy[idx][0]), int(keypoints_xy[idx][1])
            cv2.circle(img, (cx, cy), 10, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), 7, color, -1, cv2.LINE_AA)
            cv2.putText(img, str(idx), (cx + 10, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def compute_metrics(kp: np.ndarray, confs: np.ndarray) -> dict:
    metrics = {}
    def pt(i):
        return (int(kp[i][0]), int(kp[i][1])) if i < len(kp) and confs[i] > 0.3 else None

    ls, rs = pt(5), pt(6)
    lh, rh = pt(11), pt(12)

    if ls and rs:
        metrics["어깨 너비 (px)"] = abs(ls[0] - rs[0])
    if lh and rh:
        metrics["엉덩이 너비 (px)"] = abs(lh[0] - rh[0])
    if ls and rs and lh and rh:
        sy = (ls[1] + rs[1]) / 2
        hy = (lh[1] + rh[1]) / 2
        metrics["몸통 높이 (px)"] = int(abs(sy - hy))
        sw = abs(ls[0] - rs[0])
        hw = abs(lh[0] - rh[0])
        if hw > 0:
            metrics["어깨/엉덩이 비율"] = round(sw / hw, 2)
    return metrics


# ── Streamlit UI ───────────────────────────────────────────────
st.set_page_config(page_title="몸통 포즈 분석기", page_icon="🏃", layout="wide")

st.title("🏃 몸통 포즈 분석기 (Body Pose Analyzer)")
st.caption("YOLOv11 Pose를 이용해 사진에서 몸통 주요 포인트를 감지합니다.")

# ── 사이드바 ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    min_conf = st.slider("최소 감지 신뢰도", 0.1, 1.0, 0.3, 0.05)

    st.markdown("---")
    st.markdown("### 포인트 색상 범례")
    for emoji, name in [("🔵","어깨"), ("🟢","팔꿈치"), ("🟠","손목"),
                        ("🔴","엉덩이"), ("🟣","무릎"), ("🟡","코")]:
        st.write(f"{emoji} {name}")

# ── 이미지 소스 ────────────────────────────────────────────────
DEFAULT_IMAGE = Path(__file__).parent / "test.png"

uploaded = st.file_uploader(
    "다른 사진으로 변경하려면 업로드하세요 (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded is not None:
    pil_img = Image.open(uploaded).convert("RGB")
    st.caption("업로드된 이미지 사용 중")
elif DEFAULT_IMAGE.exists():
    pil_img = Image.open(DEFAULT_IMAGE).convert("RGB")
    st.caption("기본 이미지 사용 중: `test.png`")
else:
    st.warning("test.png 파일이 없습니다. 이미지를 업로드해주세요.")
    st.stop()

img_np = np.array(pil_img)

# ── 모델 로드 & 추론 ───────────────────────────────────────────
with st.spinner("모델 로딩 및 포즈 분석 중..."):
    model = load_model()
    results = model(img_np, conf=min_conf, verbose=False)

# ── 결과 표시 ──────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.subheader("📷 원본 이미지")
    st.image(pil_img, use_container_width=True)

result = results[0]
if result.keypoints is None or len(result.keypoints.xy) == 0:
    with col2:
        st.subheader("❌ 감지 실패")
        st.warning("포즈를 감지하지 못했습니다. 전신이 잘 보이는 사진을 사용해보세요.")
    st.stop()

# 첫 번째 사람 기준
kp_xy   = result.keypoints.xy[0].cpu().numpy()       # (17, 2)
kp_conf = result.keypoints.conf[0].cpu().numpy()     # (17,)

annotated = draw_pose(img_np, kp_xy, kp_conf)

with col2:
    st.subheader("✅ 분석 결과")
    st.image(annotated, use_container_width=True)

# ── 측정 지표 ──────────────────────────────────────────────────
st.markdown("---")
st.subheader("📐 주요 측정값")
metrics = compute_metrics(kp_xy, kp_conf)
if metrics:
    mcols = st.columns(len(metrics))
    for col, (name, val) in zip(mcols, metrics.items()):
        col.metric(name, val)
else:
    st.info("측정에 필요한 포인트가 충분히 감지되지 않았습니다.")

# ── 랜드마크 상세 테이블 ───────────────────────────────────────
st.markdown("---")
st.subheader("📋 랜드마크 상세 정보")

lm_data = []
for idx, (label, _, _) in KEYPOINTS.items():
    if idx < len(kp_xy):
        x, y = kp_xy[idx]
        conf = float(kp_conf[idx]) if idx < len(kp_conf) else 0.0
        lm_data.append({
            "인덱스": idx,
            "이름": label,
            "X (px)": int(x),
            "Y (px)": int(y),
            "신뢰도": round(conf, 3),
            "감지 여부": "✅" if conf > 0.5 else "⚠️",
        })

st.dataframe(lm_data, use_container_width=True, hide_index=True)

# ── 다운로드 ──────────────────────────────────────────────────
buf = io.BytesIO()
Image.fromarray(annotated).save(buf, format="PNG")
st.download_button(
    label="📥 분석 이미지 다운로드",
    data=buf.getvalue(),
    file_name="pose_analyzed.png",
    mime="image/png",
)
