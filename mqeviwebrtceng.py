""" English version of Visual Inspection Distance (WebRTC) using MediaPipe Pose and Streamlit WebRTC """
import os
import shutil
import math
import time
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
st.set_page_config(page_title="Visual Inspection Distance (WebRTC版)", layout="wide")

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
# 【替換】使用 Metered.ca 的 TURN 伺服器設定
# -----------------------------------------------------------------------------
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        # 保留 Google 免費 STUN（處理大部分普通網路環境）
        {"urls": ["stun:stun.l.google.com:19302"]},
        
        # 加入 Metered TURN 設定（處理嚴格防火牆/Symmetric NAT）
        {
            "urls": [
                "turn:openrelay.metered.ca:80",
                "turn:openrelay.metered.ca:443",
                "turns:openrelay.metered.ca:443?transport=tcp"
            ],
            "username": "9736e7593c3eaefcd10e0afb",  # 填入 Metered 帳號/API Key
            "credential": "0WH5wjRGFhfv3KxS" # 填入 Metered 密碼/Secret
        }
    ]
})

# # -----------------------------------------------------------------------------
# # 【方案 A 修改】WebRTC 多組 Google / Twilio STUN 伺服器配置（提高連線成功率）
# # -----------------------------------------------------------------------------
# RTC_CONFIG = RTCConfiguration({
#     "iceServers": [
#         {"urls": ["stun:stun.l.google.com:19302"]},
#         {"urls": ["stun:stun1.l.google.com:19302"]},
#         {"urls": ["stun:stun2.l.google.com:19302"]},
#         {"urls": ["stun:stun3.l.google.com:19302"]},
#         {"urls": ["stun:stun4.l.google.com:19302"]},
#         {"urls": ["stun:global.stun.twilio.com:3478"]}
#     ]
# })

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
        self.status_str = "No detection human body"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        status_str = "No detection human body"
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

            if self.min_target_cm <= current_dist_cm <= self.max_target_cm:
                status_str = "Pass"
                display_overlay_text = "PASS"
                line_color = (0, 255, 0)
            elif current_dist_cm < self.min_target_cm:
                diff_cm = self.min_target_cm - current_dist_cm
                status_str = f"Please move further {diff_cm:.1f} cm"
                display_overlay_text = f"TOO CLOSE (-{diff_cm:.1f}cm)"
                line_color = (0, 0, 255)
            else:
                diff_cm = current_dist_cm - self.max_target_cm
                status_str = f"Please move closer {diff_cm:.1f} cm"
                display_overlay_text = f"TOO FAR (+{diff_cm:.1f}cm)"
                line_color = (0, 0, 255)

            cv2.circle(img, (eye_mid_x, eye_mid_y), 8, (0, 255, 255), -1)
            cv2.circle(img, (thumb_mid_x, thumb_mid_y), 8, (255, 255, 0), -1)
            cv2.line(img, (eye_mid_x, eye_mid_y), (thumb_mid_x, thumb_mid_y), line_color, 3)

            cv2.putText(img, f"{current_dist_cm:.1f} cm | {display_overlay_text}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, line_color, 2)
        else:
            # 未偵測到人體時顯示提示框
            cv2.rectangle(img, (30, 30), (w - 30, h - 30), (0, 255, 255), 2)
            cv2.putText(img, "PLEASE ENTER FRAME (CENTER YOURSELF)", (50, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        self.current_dist_cm = current_dist_cm
        self.status_str = status_str

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -----------------------------------------------------------------------------
# 4. Streamlit UI 介面設定
# -----------------------------------------------------------------------------
st.title("📷 Visual Inspection Distance (WebRTC Version)")
st.caption("Standard Range: 30 ~ 32 cm")

st.sidebar.header("⚙️ System Parameters")
use_3d_world = st.sidebar.toggle("Enable 3D World Real-distance Mode", value=True)
calib_ratio = st.sidebar.slider("Distance Calibration Ratio", min_value=0.1, max_value=2.0, value=0.85, step=0.01)
scale_factor = st.sidebar.slider("2D Pixels to Centimeters Ratio (Scale Factor)", min_value=0.01, max_value=0.50, value=0.15, step=0.005)

col1, col2 = st.columns([3, 1])

with col1:
    ctx = webrtc_streamer(
        key="pose-distance-detector",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=PoseVideoProcessor,
        # 【修改點】將解析度調至 640x480 加快 ICE 握手與串流建立速度
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 30}
            },
            "audio": False
        },
        async_processing=True,
    )

if ctx.video_processor:
    ctx.video_processor.use_3d_world = use_3d_world
    ctx.video_processor.calib_ratio = calib_ratio
    ctx.video_processor.scale_factor = scale_factor

with col2:
    st.subheader("📊 Inspection result and voice prompt")
    
    # 點擊啟用語音按鈕
    components.html("""
        <button id="speech-btn" onclick="initSpeech()" style="
            width: 100%;
            padding: 12px;
            background-color: #198754;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;">
            🔊 Click here to enable voice prompt
        </button>
        <script>
            function initSpeech() {
                window.speechSynthesis.cancel();
                var msg = new SpeechSynthesisUtterance("Enable Voice prompt function");
                msg.lang = " en-US";
                window.speechSynthesis.speak(msg);
                
                var btn = document.getElementById("speech-btn");
                btn.style.backgroundColor = "#0d6efd";
                btn.innerText = "✅ Voice prompt is ready";
            }
        </script>
    """, height=55)

    # 片段自動刷新區塊（每 1.0 秒自動同步 UI 與驅動語音）
    @st.fragment(run_every=1.0)
    def render_realtime_metrics():
        if ctx.video_processor and ctx.state.playing:
            status_val = ctx.video_processor.status_str
            dist_val = ctx.video_processor.current_dist_cm

            st.metric("current inspection status", status_val)
            st.metric("measurement distance", f"{dist_val:.1f} cm")

            # 狀態改變時發聲
            if status_val not in ["No detection human body", "Please turn on the camera"]:
                safe_text = status_val.replace("'", "\\'")
                components.html(f"""
                    <script>
                        if ('speechSynthesis' in window) {{
                            window.speechSynthesis.cancel();
                            var msg = new SpeechSynthesisUtterance('{safe_text}');
                            msg.lang = 'en-US';
                            window.speechSynthesis.speak(msg);
                        }}
                    </script>
                """, height=0, width=0)
        else:
            st.metric("current inspection status", "Please turn on the camera")
            st.metric("measurement distance", "0.0 cm")

    # 片段自動刷新區塊（每 1.0 秒自動同步 UI 與驅動語音）
    @st.fragment(run_every=1.0)
    def render_realtime_metrics():
        if ctx.video_processor and ctx.state.playing:
            status_val = ctx.video_processor.status_str
            dist_val = ctx.video_processor.current_dist_cm

            st.metric("current inspection status", status_val)
            st.metric("measurement distance", f"{dist_val:.1f} cm")

            # 狀態改變時發聲
            if status_val not in ["No detection human body", "Please turn on the camera"]:
                safe_text = status_val.replace("'", "\\'")
                components.html(f"""
                    <script>
                        if ('speechSynthesis' in window) {{
                            // 僅在狀態文字改變時才觸發發音，避免每秒 cancel 導致音量變小或聲音中斷
                            if (window.parent.lastSpokenText !== '{safe_text}') {{
                                window.parent.lastSpokenText = '{safe_text}';
                                window.speechSynthesis.cancel();
                                
                                var msg = new SpeechSynthesisUtterance('{safe_text}');
                                msg.lang = 'en-US';
                                msg.pitch = 1.3;  // 提高音調 (1.0為預設，1.2~1.5偏偏高女聲)
                                msg.rate = 1.0;   // 語速
                                msg.volume = 1.0; // 音量 (最大 1.0)

                                // 優先選用瀏覽器內建的女聲
                                var voices = window.speechSynthesis.getVoices();
                                var femaleVoice = voices.find(v => 
                                    v.lang.includes('en') && 
                                    (v.name.includes('Female') || v.name.includes('Zira') || v.name.includes('Samantha') || v.name.includes('Google US English'))
                                );
                                if (femaleVoice) {{
                                    msg.voice = femaleVoice;
                                }}

                                window.speechSynthesis.speak(msg);
                            }}
                        }}
                    </script>
                """, height=0, width=0)
        else:
            st.metric("current inspection status", "Please turn on the camera")
            st.metric("measurement distance", "0.0 cm")

    render_realtime_metrics()