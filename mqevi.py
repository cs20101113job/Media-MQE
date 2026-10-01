import cv2
import numpy as np
import streamlit as st
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 1. 載入模型（改用 model_complexity=1 避免下載權限問題）
@st.cache_resource
def load_pose_model():
    return mp_pose.Pose(
        static_image_mode=True,      # 配合單張影像處理
        model_complexity=1,         # 內建模型，無需動態下載，解決 PermissionError
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

st.title("MediaPipe 姿勢辨識系統")

# 2. 使用 Streamlit 原生相機元件（替代 cv2.VideoCapture）
img_file_buffer = st.camera_input("請拍照進行姿勢分析")

if img_file_buffer is not None:
    # 將拍攝的照片轉為 OpenCV 格式
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # 載入模型並進行推論
    pose = load_pose_model()
    results = pose.process(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

    # 3. 繪製骨架骨骼線條
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            cv2_img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # 4. 顯示結果影像
    st.image(cv2_img, channels="BGR", caption="姿勢分析結果")