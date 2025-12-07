import streamlit as st
import cv2
import numpy as np
import pyttsx3
import time
from ultralytics import YOLO, RTDETR
import os

# --- Page Config ---
st.set_page_config(
    page_title="Visual Accessibility App", 
    layout="centered"
)

# --- Session State ---
if 'force_description' not in st.session_state:
    st.session_state['force_description'] = False

# --- Custom Styling ---
st.markdown("""
    <style>
    h1 { text-align: center; color: #333; margin-bottom: 0px; }
    .subtitle { text-align: center; color: #666; font-size: 16px; margin-bottom: 20px; }
    
    .caption-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6; 
        text-align: center;
        font-size: 20px;
        font-weight: 500;
        color: #000;
        margin-top: 15px;
        border: 1px solid #ddd;
    }
    
    /* Make the button big and visible */
    .stButton button {
        width: 100%;
        font-weight: bold;
        height: 60px;
        font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Configuration ---
MODELS_DIR = "models"
MODEL_OPTIONS = {
    "YOLO11n": {"file": "yolo11n.pt", "type": "YOLO"},
    "YOLO11s": {"file": "yolo11s.pt", "type": "YOLO"},
    "YOLO11m": {"file": "yolo11m.pt", "type": "YOLO"},
    "YOLOv8n": {"file": "yolo8nano.pt", "type": "YOLO"},
    "RT-DETR": {"file": "rtdetr.pt", "type": "RTDETR"} 
}

# --- Helper Functions ---
def get_position(box_coords, frame_width):
    x1, _, x2, _ = box_coords
    x_mean = (x1 + x2) / 2
    if x_mean < frame_width / 3:
        return "on the left"
    elif x_mean > frame_width * 2 / 3:
        return "on the right"
    else:
        return "in front"

@st.cache_resource
def load_model(model_name):
    model_info = MODEL_OPTIONS[model_name]
    model_path = os.path.join(MODELS_DIR, model_info["file"])
    
    if not os.path.exists(model_path):
        return None, f"Model file not found: {model_path}"
    
    try:
        if model_info["type"] == "YOLO":
            model = YOLO(model_path)
        else:
            model = RTDETR(model_path)
        return model, None
    except Exception as e:
        return None, str(e)

def speak_text(engine, text):
    if engine:
        try:
            engine.say(text)
            engine.runAndWait()
        except:
            pass

# --- Main App Logic ---
def main():
    st.title("Visual Accessibility Assistant")
    
    st.markdown("""
    <div class='subtitle'>
        The app describes objects in your environment via text and audio.
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()

    # --- Sidebar ---
    st.sidebar.title("Settings")
    
    st.sidebar.header("Model Configuration")
    selected_model_name = st.sidebar.selectbox("Select AI Model", list(MODEL_OPTIONS.keys()))
    
    st.sidebar.divider()

    st.sidebar.header("Audio Feedback")
    enable_audio = st.sidebar.checkbox("Enable Continuous Audio", value=True)
    
    with st.sidebar.expander("Advanced Audio Settings"):
        cooldown = st.slider("Audio Interval (seconds)", 2.0, 10.0, 5.0)

    st.sidebar.divider()

    with st.sidebar.container():
        run_app = st.toggle("Activate Camera (Continuous)", value=False)

    # --- MAIN CONTENT ---
    status_placeholder = st.empty()
    
    # --- 1. DESCRIBE BUTTON (ALWAYS VISIBLE) ---
    # We place this here so it is always seen, regardless of camera state.
    force_snapshot = False
    if st.button("🔊 Describe Scene Now", type="primary"):
        st.session_state['force_description'] = True
        if not run_app:
            force_snapshot = True

    # --- 2. CAMERA CONTAINER ---
    with st.container(border=True):
        st.markdown("**Live Camera Feed**")
        frame_placeholder = st.empty()
        
        # Initial State Message
        if not run_app and not force_snapshot:
            frame_placeholder.markdown(
                """
                <div style='padding: 60px; text-align: center; background-color: #f9f9f9; color: #888;'>
                    Camera is OFF. <br>
                    Click <b>Describe Scene Now</b> for a snapshot<br>
                    or toggle <b>Activate Camera</b> for continuous mode.
                </div>
                """, 
                unsafe_allow_html=True
            )

    # --- 3. SUBTITLES CONTAINER ---
    subtitle_placeholder = st.empty()

    # --- LOGIC ---
    
    # Decide if we are running the loop (Continuous) or just one frame (Snapshot)
    should_run = run_app or force_snapshot

    if should_run:
        # Load Model (Cached)
        model, error = load_model(selected_model_name)
        if error:
            status_placeholder.error(error)
            return

        class_names = model.names
        colors = np.random.uniform(0, 255, size=(len(class_names), 3))

        # Init Audio
        engine = None
        try:
            engine = pyttsx3.init()
        except:
            pass

        # Open Camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            status_placeholder.error("Cannot open webcam.")
            return

        if run_app:
            status_placeholder.success("✅ Camera Active")
        else:
            status_placeholder.info("📸 Taking Snapshot...")

        last_speech_time = 0
        
        # --- Frame Loop ---
        while cap.isOpened():
            # Stop condition
            if not run_app and not force_snapshot:
                break

            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read frame.")
                break

            # Inference
            results = model(frame, conf=0.5, verbose=False)
            descriptions = []
            
            # Process Detections
            frame_height, frame_width = frame.shape[:2]
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = coords
                    cls = int(box.cls[0].cpu().numpy())
                    
                    if cls in class_names:
                        class_name = class_names[cls]
                    else:
                        class_name = "Unknown"

                    pos = get_position(coords, frame_width)
                    descriptions.append(f"a {class_name} {pos}")

                    # Draw Box
                    color = colors[cls] if cls < len(colors) else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Show Frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # --- AUDIO/TEXT LOGIC ---
            current_time = time.time()
            
            # Triggers
            time_trigger = (current_time - last_speech_time > cooldown)
            manual_trigger = st.session_state['force_description']
            
            # Determine Sentence
            sentence = ""
            if descriptions:
                if len(descriptions) == 1:
                    sentence = f"There is {descriptions[0]}."
                elif len(descriptions) == 2:
                    sentence = f"There is {descriptions[0]} and {descriptions[1]}."
                else:
                    sentence = "There are " + ", ".join(descriptions[:2]) + "."
            elif force_snapshot:
                sentence = "No objects detected."

            # Execute Feedback
            if sentence:
                # Condition 1: Forced by Button (Snapshot or Manual Click)
                if manual_trigger:
                    subtitle_placeholder.markdown(f"<div class='caption-box'>{sentence}</div>", unsafe_allow_html=True)
                    speak_text(engine, sentence)
                    
                    st.session_state['force_description'] = False
                    last_speech_time = time.time()
                    
                    # If this was a snapshot, break the loop immediately after one pass
                    if force_snapshot:
                        force_snapshot = False 
                        break

                # Condition 2: Continuous Mode (Audio ON + Timer)
                elif run_app and enable_audio and time_trigger:
                    subtitle_placeholder.markdown(f"<div class='caption-box'>{sentence}</div>", unsafe_allow_html=True)
                    speak_text(engine, sentence)
                    last_speech_time = time.time()

                # Condition 3: Continuous Mode (Audio OFF + Timer -> Text Only)
                elif run_app and not enable_audio and time_trigger:
                     subtitle_placeholder.markdown(f"<div class='caption-box'>{sentence}</div>", unsafe_allow_html=True)
                     last_speech_time = time.time()

        # Cleanup
        cap.release()
        if not run_app:
            status_placeholder.info("Camera stopped.")
    
    else:
        subtitle_placeholder.empty()

if __name__ == "__main__":
    main()