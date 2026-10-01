import os
import cv2
import math
import time
import subprocess
import urllib.request
import numpy as np
import streamlit as st
import mediapipe as mp

# -----------------------------------------------------------------------------
# 1. 必須是第一個執行的 Streamlit 指令
# -----------------------------------------------------------------------------
st.set_page_config(page_title="目檢標準距離檢測系統", layout="wide")

# -----------------------------------------------------------------------------
# 2. 自動預下載並修正 MediaPipe 模型權限 (解決 Streamlit Cloud PermissionError)
# -----------------------------------------------------------------------------
def setup_mediapipe_model():
    """解決 Streamlit Cloud 無法寫入 site-packages 的權限問題"""
    try:
        mp_path = os.path.dirname(mp.__file__)
        target_dir = os.path.join(mp_path, "modules", "pose_landmark")
        target_file = os.path.join(target_dir, "pose_landmark_lite.tflite")
        
        os.makedirs(target_dir, exist_ok=True)
        
        try:
            os.chmod(target_dir, 0o777)
        except Exception:
            pass
            
        if not os.path.exists(target_file):
            url = "https://storage.googleapis.com/mediapipe-assets/pose_landmark_lite.tflite"
            urllib.request.urlretrieve(url, target_file)
    except Exception as e:
        print(f"MediaPipe 模型預處理提示: {e}")

# 執行模型權限修復
setup_mediapipe_model()

mp_pose = mp.solutions.pose

@st.cache_resource
def load_pose_model():
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

# -----------------------------------------------------------------------------
# 3. Session State 與 語音播報
# -----------------------------------------------------------------------------
if "last_speak_time" not in st.session_state:
    st.session_state.last_speak_time = 0
if "current_speech_proc" not in st.session_state:
    st.session_state.current_speech_proc = None
if "target_reached_spoken" not in st.session_state:
    st.session_state.target_reached_spoken = False

def speak_async(text, cooldown=2.5, force=False):
    """非同步語音播報 (適用於 macOS / Linux)"""
    current_time = time.time()
    
    if force or (current_time - st.session_state.last_speak_time >= cooldown):
        st.session_state.last_speak_time = current_time
        
        if force and st.session_state.current_speech_proc is not None:
            if st.session_state.current_speech_proc.poll() is None:
                st.session_state.current_speech_proc.terminate()
        
        try:
            # 判斷是否在 macOS 支援 say 指令
            st.session_state.current_speech_proc = subprocess.Popen(['say', text])
            return True
        except Exception:
            return False
    return False

# -----------------------------------------------------------------------------
# 4. Streamlit UI 介面設定
# -----------------------------------------------------------------------------
st.title("📷 目檢標準距離檢測系統")
st.caption("標準範圍：30 ~ 32 cm")

st.sidebar.header("⚙️ 系統參數設定")

run_camera = st.sidebar.toggle("開啟攝影機", value=True)
use_3d_world = st.sidebar.toggle("啟用 3D 空間真實距離模式", value=True)
calib_ratio = st.sidebar.slider("距離校正倍率 (Calib Ratio)", min_value=0.1, max_value=2.0, value=0.85, step=0.01)
scale_factor = st.sidebar.slider("2D 像素轉公分比例 (Scale Factor)", min_value=0.01, max_value=0.50, value=0.15, step=0.005)

if st.sidebar.button("重置校正倍率為 1.0"):
    calib_ratio = 1.0

MIN_TARGET_CM = 30.0
MAX_TARGET_CM = 32.0

col1, col2 = st.columns([3, 1])
with col1:
    image_container = st.empty()
with col2:
    status_metric = st.empty()
    dist_metric = st.empty()

# -----------------------------------------------------------------------------
# 5. 核心處理邏輯
# -----------------------------------------------------------------------------
if run_camera:
    pose = load_pose_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("⚠️ 無法開啟攝影機！若已部署至 Streamlit Cloud，雲端伺服器無法使用 OpenCV 讀取使用者本機攝影機。")
    else:
        while cap.isOpened() and run_camera:
            ret, frame = cap.read()
            if not ret:
                st.warning("無法取得攝影機畫面。")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            status_str = "未偵測到人體標記"
            line_color = (200, 200, 200)
            current_dist_cm = 0.0

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

                raw_dist_cm = 0.0

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
                    if not st.session_state.target_reached_spoken:
                        if speak_async("距離合規", force=True):
                            st.session_state.target_reached_spoken = True

                elif current_dist_cm < MIN_TARGET_CM:
                    diff_cm = MIN_TARGET_CM - current_dist_cm
                    status_str = f"TOO CLOSE (請拉遠 {diff_cm:.1f} cm)"
                    line_color = (0, 0, 255)
                    st.session_state.target_reached_spoken = False
                    speak_async(f"太近，請拉遠{diff_cm:.1f}公分", cooldown=2.5)

                else:
                    diff_cm = current_dist_cm - MAX_TARGET_CM
                    status_str = f"TOO FAR (請靠近 {diff_cm:.1f} cm)"
                    line_color = (0, 0, 255)
                    st.session_state.target_reached_spoken = False
                    speak_async(f"太遠，請靠近{diff_cm:.1f}公分", cooldown=2.5)

                cv2.circle(frame, (eye_mid_x, eye_mid_y), 8, (0, 255, 255), -1)
                cv2.circle(frame, (thumb_mid_x, thumb_mid_y), 8, (255, 255, 0), -1)
                cv2.line(frame, (eye_mid_x, eye_mid_y), (thumb_mid_x, thumb_mid_y), line_color, 3)

                cv2.putText(frame, "A (Eye Mid)", (eye_mid_x - 50, eye_mid_y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, "B (Thumb Mid)", (thumb_mid_x - 50, thumb_mid_y + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            status_metric.metric("檢測狀態", status_str)
            dist_metric.metric("當前測量距離", f"{current_dist_cm:.1f} cm")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                image_container.image(frame_rgb, channels="RGB", use_container_width=True)
            except Exception:
                image_container.image(frame_rgb, channels="RGB")

            time.sleep(0.01)

        cap.release()