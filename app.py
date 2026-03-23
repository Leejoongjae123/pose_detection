import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
import io

# ── MediaPipe 초기화 ──────────────────────────────────────────
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ── 관심 포인트 정의 (몸통 주요 포인트) ────────────────────────
TORSO_LANDMARKS = {
    0:  ("코 (Nose)",            (255, 200, 0)),
    11: ("왼쪽 어깨 (L Shoulder)", (0, 120, 255)),
    12: ("오른쪽 어깨 (R Shoulder)",(0, 80, 200)),
    13: ("왼쪽 팔꿈치 (L Elbow)",  (0, 200, 80)),
    14: ("오른쪽 팔꿈치 (R Elbow)",(0, 160, 60)),
    15: ("왼쪽 손목 (L Wrist)",   (255, 140, 0)),
    16: ("오른쪽 손목 (R Wrist)",  (200, 100, 0)),
    23: ("왼쪽 엉덩이 (L Hip)",   (200, 0, 80)),
    24: ("오른쪽 엉덩이 (R Hip)",  (160, 0, 60)),
    25: ("왼쪽 무릎 (L Knee)",    (140, 0, 200)),
    26: ("오른쪽 무릎 (R Knee)",   (100, 0, 160)),
}

# ── 연결선 정의 ────────────────────────────────────────────────
CONNECTIONS = [
    (11, 12),  # 어깨 연결
    (11, 13),  # 왼 어깨-팔꿈치
    (13, 15),  # 왼 팔꿈치-손목
    (12, 14),  # 오른 어깨-팔꿈치
    (14, 16),  # 오른 팔꿈치-손목
    (11, 23),  # 왼 어깨-엉덩이
    (12, 24),  # 오른 어깨-엉덩이
    (23, 24),  # 엉덩이 연결
    (23, 25),  # 왼 엉덩이-무릎
    (24, 26),  # 오른 엉덩이-무릎
    (0, 11),   # 코-왼 어깨
    (0, 12),   # 코-오른 어깨
]


def analyze_pose(image_rgb: np.ndarray):
    """MediaPipe로 포즈 분석 수행"""
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5,
    ) as pose:
        results = pose.process(image_rgb)
    return results


def draw_landmarks(image_rgb: np.ndarray, landmarks) -> np.ndarray:
    """커스텀 색상으로 랜드마크와 연결선 그리기"""
    img = image_rgb.copy()
    h, w = img.shape[:2]

    lm = landmarks.landmark

    # 연결선 먼저 그리기
    for (a, b) in CONNECTIONS:
        if (lm[a].visibility > 0.3 and lm[b].visibility > 0.3):
            x1, y1 = int(lm[a].x * w), int(lm[a].y * h)
            x2, y2 = int(lm[b].x * w), int(lm[b].y * h)
            cv2.line(img, (x1, y1), (x2, y2), (200, 200, 200), 2, cv2.LINE_AA)

    # 랜드마크 점 그리기
    for idx, (label, color) in TORSO_LANDMARKS.items():
        lmk = lm[idx]
        if lmk.visibility > 0.3:
            cx, cy = int(lmk.x * w), int(lmk.y * h)
            # 외곽 흰 원
            cv2.circle(img, (cx, cy), 10, (255, 255, 255), -1, cv2.LINE_AA)
            # 색상 내부 원
            cv2.circle(img, (cx, cy), 7, color, -1, cv2.LINE_AA)
            # 인덱스 번호
            cv2.putText(img, str(idx), (cx + 10, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def compute_metrics(landmarks, img_w: int, img_h: int) -> dict:
    """어깨 너비, 엉덩이 너비, 몸통 높이 계산 (픽셀)"""
    lm = landmarks.landmark

    def px(idx):
        return (int(lm[idx].x * img_w), int(lm[idx].y * img_h))

    metrics = {}

    ls, rs = px(11), px(12)
    lh, rh = px(23), px(24)

    metrics["어깨 너비 (px)"] = int(abs(ls[0] - rs[0]))
    metrics["엉덩이 너비 (px)"] = int(abs(lh[0] - rh[0]))

    shoulder_mid_y = (ls[1] + rs[1]) / 2
    hip_mid_y = (lh[1] + rh[1]) / 2
    metrics["몸통 높이 (px)"] = int(abs(shoulder_mid_y - hip_mid_y))

    if metrics["엉덩이 너비 (px)"] > 0:
        metrics["어깨/엉덩이 비율"] = round(
            metrics["어깨 너비 (px)"] / metrics["엉덩이 너비 (px)"], 2
        )
    return metrics


# ── Streamlit UI ───────────────────────────────────────────────
st.set_page_config(
    page_title="몸통 포즈 분석기",
    page_icon="🏃",
    layout="wide",
)

st.title("🏃 몸통 포즈 분석기 (Body Pose Analyzer)")
st.caption("MediaPipe Pose를 이용해 사진에서 몸통 주요 포인트를 감지합니다.")

# ── 사이드바 ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    min_conf = st.slider("최소 감지 신뢰도", 0.1, 1.0, 0.5, 0.05)
    show_all = st.checkbox("모든 MediaPipe 랜드마크 표시", value=False)

    st.markdown("---")
    st.markdown("### 포인트 색상 범례")
    legend = {
        "🔵 어깨": "파란색",
        "🟢 팔꿈치": "초록색",
        "🟠 손목": "주황색",
        "🔴 엉덩이": "빨간색",
        "🟣 무릎": "보라색",
        "🟡 코": "노란색",
    }
    for k, v in legend.items():
        st.write(f"{k}: {v}")

# ── 파일 업로드 (기본: test.png) ──────────────────────────────
import os
from pathlib import Path

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
    st.caption(f"기본 이미지 사용 중: `test.png`")
else:
    st.warning("test.png 파일이 없습니다. 이미지를 업로드해주세요.")
    st.stop()

# ── 이미지 로드 ────────────────────────────────────────────────
img_np = np.array(pil_img)
h, w = img_np.shape[:2]

# ── 분석 실행 ──────────────────────────────────────────────────
with st.spinner("포즈 분석 중..."):
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=min_conf,
    ) as pose:
        results = pose.process(img_np)

# ── 결과 표시 ──────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 원본 이미지")
    st.image(pil_img, use_container_width=True)

if results.pose_landmarks is None:
    with col2:
        st.subheader("❌ 감지 실패")
        st.warning("포즈를 감지하지 못했습니다. 전신이 잘 보이는 사진을 사용하거나 신뢰도를 낮춰보세요.")
    st.stop()

# 랜드마크 그리기
annotated = draw_landmarks(img_np, results.pose_landmarks)

if show_all:
    # MediaPipe 기본 드로잉 추가
    ann_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        ann_bgr,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(180, 180, 180), thickness=1, circle_radius=2),
        mp_drawing.DrawingSpec(color=(100, 100, 100), thickness=1),
    )
    annotated = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)

with col2:
    st.subheader("✅ 분석 결과")
    st.image(annotated, use_container_width=True)

# ── 측정 지표 ──────────────────────────────────────────────────
st.markdown("---")
st.subheader("📐 주요 측정값")

metrics = compute_metrics(results.pose_landmarks, w, h)
mcols = st.columns(len(metrics))
for col, (name, val) in zip(mcols, metrics.items()):
    col.metric(name, val)

# ── 랜드마크 상세 테이블 ───────────────────────────────────────
st.markdown("---")
st.subheader("📋 랜드마크 상세 정보")

lm_data = []
for idx, (label, _) in TORSO_LANDMARKS.items():
    lmk = results.pose_landmarks.landmark[idx]
    lm_data.append({
        "인덱스": idx,
        "이름": label,
        "X (비율)": round(lmk.x, 4),
        "Y (비율)": round(lmk.y, 4),
        "Z (깊이)": round(lmk.z, 4),
        "가시성": round(lmk.visibility, 3),
        "감지 여부": "✅" if lmk.visibility > 0.5 else "⚠️",
    })

st.dataframe(lm_data, use_container_width=True, hide_index=True)

# ── 분석 이미지 다운로드 ──────────────────────────────────────
buf = io.BytesIO()
Image.fromarray(annotated).save(buf, format="PNG")
st.download_button(
    label="📥 분석 이미지 다운로드",
    data=buf.getvalue(),
    file_name="pose_analyzed.png",
    mime="image/png",
)
