import math
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import streamlit.components.v1 as components

# 1. 頁面設定
st.set_page_config(page_title="目檢標準距離檢測系統", layout="wide")

# 2. 載入 MediaPipe 模型 (model_complexity=1 使用內建模型，避免權限與下載問題)
mp_pose = mp.solutions.pose

@st.cache_resource
def load_pose_model():
    return mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

# 3. 瀏覽器端 Web Speech API 語音發聲函式
def speak_in_browser(text: str):
    """透過 JavaScript 調用使用者瀏覽器內建語音合成"""
    js_code = f"""
    <script>
        var msg = new SpeechSynthesisUtterance("{text}");
        msg.lang = "zh-TW";
        window.speechSynthesis.speak(msg);
    </script>
    """
    components.html(js_code, height=0)

# 4. Streamlit UI 介面設定
st.title("📷 目檢標準距離檢測系統")
st.caption("標準範圍：30 ~ 32 cm")

st.sidebar.header("⚙️ 系統參數設定")
use_3d_world = st.sidebar.toggle("啟用 3D 空間真實距離模式", value=True)
calib_ratio = st.sidebar.slider("距離校正倍率 (Calib Ratio)", min_value=0.1, max_value=2.0, value=0.85, step=0.01)
scale_factor = st.sidebar.slider("2D 像素轉公分比例 (Scale Factor)", min_value=0.01, max_value=0.50, value=0.15, step=0.005)

MIN_TARGET_CM = 30.0
MAX_TARGET_CM = 32.0

col1, col2 = st.columns([3, 1])

with col1:
    # 使用 Streamlit 相機元件調用用戶端鏡頭
    img_file = st.camera_input("請對準鏡頭進行距離檢測")

if img_file is not None:
    # 讀取影像資料
    bytes_data = img_file.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # MediaPipe 推論
    pose = load_pose_model()
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    status_str = "未偵測到人體標記"
    line_color = (200, 200, 200)
    current_dist_cm = 0.0
    speech_text = ""

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
        right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
        eye_mid_x = int((left_eye.x + right_eye.x) / 2 * w)
        eye_mid_y = int((left_eye.y + right_eye.y) / 2 * h)

        left_thumb = landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value]
        right_thumb = landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value]
        thumb_mid_x = int((left_thumb.x + right_thumb.x) / 2 * w)
        thumb_mid_y = int((left_thumb.y + right_thumb.y) / 2 * h)

        if use_3d_world and results.pose_world_landmarks:
            wl = results.pose_world_landmarks.landmark
            l_eye_w, r_eye_w = wl[mp_pose.PoseLandmark.LEFT_EYE.value], wl[mp_pose.PoseLandmark.RIGHT_EYE.value]
            l_thumb_w, r_thumb_w = wl[mp_pose.PoseLandmark.LEFT_THUMB.value], wl[mp_pose.PoseLandmark.RIGHT_THUMB.value]

            eye_w_mid = ((l_eye_w.x + r_eye_w.x) / 2, (l_eye_w.y + r_eye_w.y) / 2, (l_eye_w.z + r_eye_w.z) / 2)
            thumb_w_mid = ((l_thumb_w.x + r_thumb_w.x) / 2, (l_thumb_w.y + r_thumb_w.y) / 2, (l_thumb_w.z + r_thumb_w.z) / 2)

            dist_meters = math.sqrt(
                (eye_w_mid[0] - thumb_w_mid[0])**2 +
                (eye_w_mid[1] - thumb_w_mid[1])**2 +
                (eye_w_mid[2] - thumb_w_mid[2])**2
            )
            raw_dist_cm = dist_meters * 100.0
        else:
            pixel_dist = math.hypot(eye_mid_x - thumb_mid_x, eye_mid_y - thumb_mid_y)
            raw_dist_cm = pixel_dist * scale_factor

        current_dist_cm = raw_dist_cm * calib_ratio

        if MIN_TARGET_CM <= current_dist_cm <= MAX_TARGET_CM:
            status_str = "PASS (合格)"
            line_color = (0, 255, 0)
            speech_text = "距離合規"
        elif current_dist_cm < MIN_TARGET_CM:
            diff_cm = MIN_TARGET_CM - current_dist_cm
            status_str = f"TOO CLOSE (請拉遠 {diff_cm:.1f} cm)"
            line_color = (0, 0, 255)
            speech_text = f"太近，請拉遠{diff_cm:.1f}公分"
        else:
            diff_cm = current_dist_cm - MAX_TARGET_CM
            status_str = f"TOO FAR (請靠近 {diff_cm:.1f} cm)"
            line_color = (0, 0, 255)
            speech_text = f"太遠，請靠近{diff_cm:.1f}公分"

        # 標註標記點與連線
        cv2.circle(frame, (eye_mid_x, eye_mid_y), 8, (0, 255, 255), -1)
        cv2.circle(frame, (thumb_mid_x, thumb_mid_y), 8, (255, 255, 0), -1)
        cv2.line(frame, (eye_mid_x, eye_mid_y), (thumb_mid_x, thumb_mid_y), line_color, 3)

        cv2.putText(frame, "A (Eye Mid)", (eye_mid_x - 50, eye_mid_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "B (Thumb Mid)", (thumb_mid_x - 50, thumb_mid_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    with col2:
        st.metric("檢測狀態", status_str)
        st.metric("當前測量距離", f"{current_dist_cm:.1f} cm")

    # 顯示分析圖檔
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

    # 觸發瀏覽器端播報
    if speech_text:
        speak_in_browser(speech_text)