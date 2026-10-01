import os
import shutil
import math
import urllib.request
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import mediapipe as mp
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

from mediapipe.python._framework_bindings import resource_util
from mediapipe.python.solutions import download_utils

# -----------------------------------------------------------------------------
# 1. 頁面配置
# -----------------------------------------------------------------------------
st.set_page_config(page_title="目檢標準距離檢測系統 (WebRTC版)", layout="wide")

# -----------------------------------------------------------------------------
# 2. 修復 MediaPipe Cloud 唯讀權限問題
# -----------------------------------------------------------------------------
def setup_mediapipe_cloud():
    mp_path = os.path.dirname(mp.__file__)
    tmp_root = "/tmp/mediapipe_root"
    tmp_mp_dir = os.path.join(tmp_root, "mediapipe")
    target_file = os.path.join(tmp_mp_dir, "modules", "pose_landmark", "pose_landmark_lite.tflite")

    if not os.path.exists(target_file):
        for root, dirs, files in os.walk(mp_path):
            rel_path = os.path.relpath(root, mp_path)
            dest_dir = os.path.join(tmp_mp_dir, rel_path) if rel_path != "." else tmp_mp_dir
            os.makedirs(dest_dir, exist_ok=True)
            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dest_dir, file)
                if not os.path.exists(dst_file):
                    try:
                        os.symlink(src_file, dst_file)
                    except Exception:
                        shutil.copy2(src_file, dst_file)

        url = "https://storage.googleapis.com/mediapipe-assets/pose_landmark_lite.tflite"
        urllib.request.urlretrieve(url, target_file)

    resource_util.set_resource_dir(tmp_root)
    download_utils.download_oss_model = lambda path: None

setup_mediapipe_cloud()
mp_pose = mp.solutions.pose

# -----------------------------------------------------------------------------
# 強化版 WebRTC RTCConfiguration (包含多組 STUN 與公共 TURN 轉發伺服器)
# -----------------------------------------------------------------------------
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:global.stun.twilio.com:3478"]},
        # 使用免費公共 TURN 伺服器穿透嚴格防火牆/NAT
        {
            "urls": ["turn:openrelay.metered.ca:80", "turn:openrelay.metered.ca:443"],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        },
        {
            "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
            "username": "openrelayproject",
            "credential": "openrelayproject"
        }
    ]
})

# -----------------------------------------------------------------------------
# 3. WebRTC 影像處理類別
# -----------------------------------------------------------------------------
class PoseVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.use_3d_world = True
        self.calib_ratio = 0.85
        self.scale_factor = 0.15
        self.min_target_cm = 30.0
        self.max_target_cm = 32.0

        self.current_dist_cm = 0.0
        self.status_str = "未偵測到人體"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        status_str = "未偵測到人體"
        line_color = (200, 200, 200)
        current_dist_cm = 0.0
        display_overlay_text = "NO DETECTION"

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

            # 判定狀態：影像畫面上只繪製英文字串 (避免 OpenCV 產生 ???? 亂碼)
            if self.min_target_cm <= current_dist_cm <= self.max_target_cm:
                status_str = "合格"
                display_overlay_text = "PASS"
                line_color = (0, 255, 0)
            elif current_dist_cm < self.min_target_cm:
                diff_cm = self.min_target_cm - current_dist_cm
                status_str = f"請拉遠 {diff_cm:.1f} 公分"
                display_overlay_text = f"TOO CLOSE (-{diff_cm:.1f}cm)"
                line_color = (0, 0, 255)
            else:
                diff_cm = current_dist_cm - self.max_target_cm
                status_str = f"請靠近 {diff_cm:.1f} 公分"
                display_overlay_text = f"TOO FAR (+{diff_cm:.1f}cm)"
                line_color = (0, 0, 255)

            cv2.circle(img, (eye_mid_x, eye_mid_y), 8, (0, 255, 255), -1)
            cv2.circle(img, (thumb_mid_x, thumb_mid_y), 8, (255, 255, 0), -1)
            cv2.line(img, (eye_mid_x, eye_mid_y), (thumb_mid_x, thumb_mid_y), line_color, 3)

            cv2.putText(img, f"{current_dist_cm:.1f} cm | {display_overlay_text}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, line_color, 2)

        self.current_dist_cm = current_dist_cm
        self.status_str = status_str

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -----------------------------------------------------------------------------
# 4. Streamlit UI 介面設定
# -----------------------------------------------------------------------------
st.title("📷 目檢標準距離檢測系統 (WebRTC 版)")
st.caption("標準範圍：30 ~ 32 cm")

st.sidebar.header("⚙️ 系統參數設定")
use_3d_world = st.sidebar.toggle("啟用 3D 空間真實距離模式", value=True)
calib_ratio = st.sidebar.slider("距離校正倍率 (Calib Ratio)", min_value=0.1, max_value=2.0, value=0.85, step=0.01)
scale_factor = st.sidebar.slider("2D 像素轉公分比例 (Scale Factor)", min_value=0.01, max_value=0.50, value=0.15, step=0.005)

col1, col2 = st.columns([3, 1])

with col1:
    ctx = webrtc_streamer(
        key="pose-distance-detector",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=PoseVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

if ctx.video_processor:
    ctx.video_processor.use_3d_world = use_3d_world
    ctx.video_processor.calib_ratio = calib_ratio
    ctx.video_processor.scale_factor = scale_factor

with col2:
    st.subheader("📊 檢測結果與語音提示")
    
    current_status = ctx.video_processor.status_str if ctx.video_processor else "請開啟攝影機"
    current_dist = f"{ctx.video_processor.current_dist_cm:.1f}" if ctx.video_processor else "0.0"

    # 使用純前端 JS 控管語音與顯示，不鎖死 Streamlit 伺服器
    components.html(f"""
        <div style="font-family: sans-serif; background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e9ecef;">
            <p style="margin: 0; font-size: 13px; color: #6c757d;">當前狀態：</p>
            <h3 style="margin: 5px 0 15px 0; color: #0d6efd;">{current_status}</h3>
            
            <p style="margin: 0; font-size: 13px; color: #6c757d;">測量距離：</p>
            <h2 style="margin: 5px 0 15px 0; color: #212529;">{current_dist} cm</h2>

            <button id="speech-btn" onclick="initSpeech()" style="
                width: 100%;
                padding: 10px;
                background-color: #198754;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
                cursor: pointer;">
                🔊 點此解鎖語音功能
            </button>
            <p id="speech-info" style="margin: 5px 0 0 0; font-size: 11px; color: #6c757d; text-align: center;">
                瀏覽器預設阻擋自動發音，開鏡頭前請先點擊一次
            </p>
        </div>

        <script>
            var isAudioUnlocked = false;
            var lastSpokenText = "";
            var lastSpeakTime = 0;

            function initSpeech() {{
                window.speechSynthesis.cancel();
                var msg = new SpeechSynthesisUtterance("語音提示功能已啟用");
                msg.lang = "zh-TW";
                window.speechSynthesis.speak(msg);
                
                isAudioUnlocked = true;
                var btn = document.getElementById("speech-btn");
                btn.style.backgroundColor = "#0d6efd";
                btn.innerText = "✅ 語音播報已就緒";
                document.getElementById("speech-info").innerText = "檢測狀態改變時將自動進行語音提醒";
            }}

            function speak(text) {{
                if (!isAudioUnlocked || !text || text === "未偵測到人體" || text === "請開啟攝影機") return;
                
                var now = Date.now();
                // 狀態改變或冷卻時間滿 2.5 秒時發聲，防止聲音疊加
                if (text !== lastSpokenText || (now - lastSpeakTime) > 2500) {{
                    window.speechSynthesis.cancel();
                    var msg = new SpeechSynthesisUtterance(text);
                    msg.lang = "zh-TW";
                    msg.rate = 1.0;
                    window.speechSynthesis.speak(msg);
                    
                    lastSpokenText = text;
                    lastSpeakTime = now;
                }}
            }}

            // 自動嘗試播放當前文字
            speak("{current_status}");
        </script>
    """, height=220)