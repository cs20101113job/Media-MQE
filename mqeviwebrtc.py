import os
import math
import time
import urllib.request
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

# -----------------------------------------------------------------------------
# 1. 頁面配置 (必須是第一個 Streamlit 指令)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="目檢標準距離檢測系統 (WebRTC版)", layout="wide")

# -----------------------------------------------------------------------------
# 2. 自動預下載並修正 MediaPipe 模型權限 (解決 Streamlit Cloud 權限錯誤)
# -----------------------------------------------------------------------------
def setup_mediapipe_model():
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

setup_mediapipe_model()

mp_pose = mp.solutions.pose

# 設定 STUN 伺服器 (確保雲端部署時能夠穿透 NAT 建立 WebRTC 連線)
RTC_CONFIG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# -----------------------------------------------------------------------------
# 3. WebRTC 影像處理類別 (VideoProcessorBase)
# -----------------------------------------------------------------------------
class PoseVideoProcessor(VideoProcessorBase):
    def __init__(self):
        # 在獨立線程中初始化 MediaPipe Pose 模型
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # 動態控制參數 (可由 UI 傳入更新)
        self.use_3d_world = True
        self.calib_ratio = 0.85
        self.scale_factor = 0.15
        self.min_target_cm = 30.0
        self.max_target_cm = 32.0

        # 當前狀態紀錄
        self.current_dist_cm = 0.0
        self.status_str = "未偵測到人體標記"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # 將輸入的 WebRTC 畫面轉換為 OpenCV 格式 (BGR)
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # 水平鏡像
        h, w, _ = img.shape
        
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        status_str = "未偵測到人體標記"
        line_color = (200, 200, 200)
        current_dist_cm = 0.0

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # A. 兩眼中心點
            left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
            right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
            eye_mid_x = int((left_eye.x + right_eye.x) / 2 * w)
            eye_mid_y = int((left_eye.y + right_eye.y) / 2 * h)

            # B. 雙手大拇指指尖中心點
            left_thumb = landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value]
            right_thumb = landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value]
            thumb_mid_x = int((left_thumb.x + right_thumb.x) / 2 * w)
            thumb_mid_y = int((left_thumb.y + right_thumb.y) / 2 * h)

            # C. 距離計算
            raw_dist_cm = 0.0

            if self.use_3d_world and results.pose_world_landmarks:
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
                raw_dist_cm = pixel_dist * self.scale_factor

            current_dist_cm = raw_dist_cm * self.calib_ratio

            # 判斷標準距離
            if self.min_target_cm <= current_dist_cm <= self.max_target_cm:
                status_str = "PASS (合格)"
                line_color = (0, 255, 0)
            elif current_dist_cm < self.min_target_cm:
                diff_cm = self.min_target_cm - current_dist_cm
                status_str = f"TOO CLOSE (請拉遠 {diff_cm:.1f} cm)"
                line_color = (0, 0, 255)
            else:
                diff_cm = current_dist_cm - self.max_target_cm
                status_str = f"TOO FAR (請靠近 {diff_cm:.1f} cm)"
                line_color = (0, 0, 255)

            # 繪製視覺化特徵標記
            cv2.circle(img, (eye_mid_x, eye_mid_y), 8, (0, 255, 255), -1)
            cv2.circle(img, (thumb_mid_x, thumb_mid_y), 8, (255, 255, 0), -1)
            cv2.line(img, (eye_mid_x, eye_mid_y), (thumb_mid_x, thumb_mid_y), line_color, 3)

            cv2.putText(img, "A (Eye Mid)", (eye_mid_x - 50, eye_mid_y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(img, "B (Thumb Mid)", (thumb_mid_x - 50, thumb_mid_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # 將即時結果直接繪製在畫面上，提升即時感
            cv2.putText(img, f"Dist: {current_dist_cm:.1f} cm | {status_str}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, line_color, 2)

        self.current_dist_cm = current_dist_cm
        self.status_str = status_str

        # 將處理後的影像幀轉回 WebRTC 輸出格式
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -----------------------------------------------------------------------------
# 4. Streamlit UI 介面設定
# -----------------------------------------------------------------------------
st.title("📷 目檢標準距離檢測系統 (WebRTC 版)")
st.caption("標準範圍：30 ~ 32 cm")

# 側邊欄控制項
st.sidebar.header("⚙️ 系統參數設定")

use_3d_world = st.sidebar.toggle("啟用 3D 空間真實距離模式", value=True)
calib_ratio = st.sidebar.slider("距離校正倍率 (Calib Ratio)", min_value=0.1, max_value=2.0, value=0.85, step=0.01)
scale_factor = st.sidebar.slider("2D 像素轉公分比例 (Scale Factor)", min_value=0.01, max_value=0.50, value=0.15, step=0.005)

if st.sidebar.button("重置校正倍率為 1.0"):
    calib_ratio = 1.0

col1, col2 = st.columns([3, 1])

with col1:
    # 建立 WebRTC 串流組件
    ctx = webrtc_streamer(
        key="pose-distance-detector",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=PoseVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# 即時將 Streamlit 控制項變數同步至 WebRTC 處理線程
if ctx.video_processor:
    ctx.video_processor.use_3d_world = use_3d_world
    ctx.video_processor.calib_ratio = calib_ratio
    ctx.video_processor.scale_factor = scale_factor

with col2:
    status_metric = st.empty()
    dist_metric = st.empty()
    
    if ctx.video_processor:
        status_metric.metric("檢測狀態", ctx.video_processor.status_str)
        dist_metric.metric("當前測量距離", f"{ctx.video_processor.current_dist_cm:.1f} cm")
    else:
        status_metric.metric("檢測狀態", "請點擊 START 開啟攝影機")
        dist_metric.metric("當前測量距離", "0.0 cm")