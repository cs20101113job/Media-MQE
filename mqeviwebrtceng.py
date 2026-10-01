# -----------------------------------------------------------------------------
# 4. Streamlit UI 介面設定
# -----------------------------------------------------------------------------
st.title("📷 Visual Inspection Distance (WebRTC Version)")
st.caption("Standard Range: 30 ~ 32 cm")

# Session State 初始化：儲存上次播報的狀態分類與時間戳記
if "last_speech_category" not in st.session_state:
    st.session_state.last_speech_category = ""
if "last_speech_time" not in st.session_state:
    st.session_state.last_speech_time = 0.0

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
    
    # 啟用語音按鈕（加入人聲選擇與音調設定）
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
                msg.lang = "en-US";
                msg.pitch = 1.1; // 略微提升音調，讓聲音更自然
                msg.rate = 1.0;  // 正常語速

                var voices = window.speechSynthesis.getVoices();
                // 優先選擇常見自然的英文女聲/標準語音
                var preferredVoice = voices.find(v => v.lang.includes('en') && (
                    v.name.includes('Google') || 
                    v.name.includes('Zira') || 
                    v.name.includes('Samantha') || 
                    v.name.includes('Jenny') ||
                    v.name.includes('Natural')
                ));
                if (preferredVoice) msg.voice = preferredVoice;

                window.speechSynthesis.speak(msg);
                
                var btn = document.getElementById("speech-btn");
                btn.style.backgroundColor = "#0d6efd";
                btn.innerText = "✅ Voice prompt is ready";
            }
        </script>
    """, height=55)

    # 每秒刷新並判斷語音觸發
    @st.fragment(run_every=1.0)
    def render_realtime_metrics():
        if ctx.video_processor and ctx.state.playing:
            status_val = ctx.video_processor.status_str
            category_val = getattr(ctx.video_processor, "status_category", "NO_HUMAN")
            dist_val = ctx.video_processor.current_dist_cm

            st.metric("current inspection status", status_val)
            st.metric("measurement distance", f"{dist_val:.1f} cm")

            current_time = time.time()
            time_passed = current_time - st.session_state.last_speech_time

            # 語音防抖條件：
            # 1. 偵測到有效人體 (類別非 NO_HUMAN)
            # 2. 狀態分類發生改變 OR 距離上次發聲已滿 3 秒 (Cooldown)
            if category_val != "NO_HUMAN":
                if category_val != st.session_state.last_speech_category or time_passed >= 3.0:
                    st.session_state.last_speech_category = category_val
                    st.session_state.last_speech_time = current_time

                    # 固定語意對應
                    speech_text_map = {
                        "PASS": "Pass",
                        "TOO_CLOSE": "Please move back",
                        "TOO_FAR": "Please move closer"
                    }
                    speech_text = speech_text_map.get(category_val, "")

                    if speech_text:
                        components.html(f"""
                            <script>
                                if ('speechSynthesis' in window) {{
                                    window.speechSynthesis.cancel();
                                    var msg = new SpeechSynthesisUtterance('{speech_text}');
                                    msg.lang = 'en-US';
                                    msg.pitch = 1.1; // 調整音調（1.0~1.2 之間最自然）
                                    msg.rate = 1.0;  // 語速

                                    function speakWithSelectedVoice() {{
                                        var voices = window.speechSynthesis.getVoices();
                                        // 優先篩選清晰自然的英語人聲
                                        var selectedVoice = voices.find(v => 
                                            v.lang.startsWith('en') && (
                                                v.name.includes('Google') || 
                                                v.name.includes('Zira') || 
                                                v.name.includes('Samantha') || 
                                                v.name.includes('Jenny') ||
                                                v.name.includes('Natural') ||
                                                v.name.includes('Female')
                                            )
                                        ) || voices.find(v => v.lang.startsWith('en'));

                                        if (selectedVoice) {{
                                            msg.voice = selectedVoice;
                                        }}
                                        window.speechSynthesis.speak(msg);
                                    }}

                                    // Chrome/Edge 異步加載語音處理
                                    var voices = window.speechSynthesis.getVoices();
                                    if (voices.length > 0) {{
                                        speakWithSelectedVoice();
                                    }} else {{
                                        window.speechSynthesis.onvoiceschanged = speakWithSelectedVoice;
                                    }}
                                }}
                            </script>
                        """, height=0, width=0)
        else:
            st.metric("current inspection status", "Please turn on the camera")
            st.metric("measurement distance", "0.0 cm")
            # 關閉攝影機時重置狀態
            st.session_state.last_speech_category = ""
            st.session_state.last_speech_time = 0.0

    render_realtime_metrics()