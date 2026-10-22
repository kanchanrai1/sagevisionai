
from collections import Counter
from io import BytesIO

import cv2
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Object Detector",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# SAGE GREEN + CREAM THEME
# ==================================================

st.markdown("""
<style>
    :root {
        --cream: #F5F0E6;
        --sage-light: #DCE5D7;
        --sage: #718873;
        --sage-dark: #465D49;
        --rose: #C9A9A0;
        --ink: #303D35;
        --white: #FFFFFF;
    }

    .stApp {
        background-color: #F5F0E6;
        color: #303D35;
    }

    [data-testid="stHeader"] {
        background-color: rgba(245, 240, 230, 0.95);
    }

    [data-testid="stSidebar"] {
        background-color: #E5EBDD;
        border-right: 1px solid #C8D4C2;
    }

    [data-testid="stSidebar"] * {
        color: #303D35;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(125deg, #718873, #465D49);
        color: white;
        padding: 32px;
        border-radius: 24px;
        margin-bottom: 26px;
        box-shadow: 0 8px 24px rgba(70, 93, 73, 0.16);
    }

    .hero h1 {
        color: white;
        font-size: clamp(27px, 4vw, 42px);
        font-weight: 750;
        margin: 0 0 10px 0;
    }

    .hero p {
        color: #F5F0E6;
        font-size: 16px;
        margin: 0;
        line-height: 1.7;
    }

    .eyebrow {
        color: #E1E9D9;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .section-heading {
        font-size: 23px;
        font-weight: 750;
        color: #465D49;
        margin-top: 14px;
        margin-bottom: 12px;
    }

    .soft-card {
        background: #FFFFFF;
        border: 1px solid #DCE5D7;
        padding: 20px;
        border-radius: 18px;
        margin-bottom: 12px;
    }

    .soft-note {
        background: #E5EBDD;
        border-left: 4px solid #718873;
        padding: 14px 16px;
        border-radius: 8px;
        color: #303D35;
        font-size: 14px;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #DCE5D7;
        padding: 18px;
        border-radius: 16px;
        box-shadow: 0 3px 10px rgba(48, 61, 53, 0.04);
    }

    div[data-testid="stMetricLabel"] {
        color: #718873;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #303D35;
        font-weight: 750;
    }

    .stButton > button {
        background: #718873;
        color: white;
        border: 1px solid #718873;
        border-radius: 12px;
        min-height: 46px;
        font-weight: 650;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: #465D49;
        color: white;
        border-color: #465D49;
        transform: translateY(-1px);
    }

    .stDownloadButton > button {
        background: #465D49;
        color: white;
        border-radius: 12px;
        border: none;
        min-height: 48px;
        font-weight: 650;
    }

    .stDownloadButton > button:hover {
        background: #718873;
        color: white;
    }

    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 1px dashed #8CA18A;
        border-radius: 16px;
        padding: 12px;
    }

    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #718873;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #DCE5D7;
        border-radius: 12px;
        overflow: hidden;
    }

    .footer {
        color: #718873;
        text-align: center;
        font-size: 13px;
        padding: 18px 0;
        line-height: 1.8;
    }

    @media (max-width: 700px) {
        .hero {
            padding: 22px;
            border-radius: 18px;
        }

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.caption("AI Object Detection Studio")
    st.divider()

    st.markdown("###  Your workspace")
    page = st.radio(
        "Navigate",
        [" Home", " About", " Technologies"],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("###  Project goal")
    st.write(
        "Turn an ordinary image into useful information "
        "with AI-powered object detection."
    )

    st.divider()
    st.caption("Made with Python, Streamlit & YOLO")


# ==================================================
# HEADER
# ==================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">YOUR LITTLE AI STUDIO 🌱</div>
    <h1>See the world differently.</h1>
    <p>
        Upload a photo, let AI discover what's inside,
        explore object counts and confidence scores,
        and save your detection results.
    </p>
</div>
""", unsafe_allow_html=True)


# ==================================================
# ABOUT PAGE
# ==================================================

if page == "ℹ️ About":
    st.markdown(
        '<div class="section-heading">About this project</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Sage Vision is an image object detection application "
        "built with Python, Streamlit, and the YOLO model."
    )

    st.markdown("""
    <div class="soft-card">
        <h3>🌿 What can it do?</h3>
        <p>📷 Upload JPG, JPEG, and PNG images.</p>
        <p>🎯 Detect supported objects and draw bounding boxes.</p>
        <p>📊 Display confidence scores and object counts.</p>
        <p>⬇️ Download the annotated image as a PNG file.</p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "The system detects objects supported by the selected "
        "pretrained model. It does not recognize every possible object."
    )

    st.stop()


# ==================================================
# TECHNOLOGIES PAGE
# ==================================================

if page == "🛠️ Technologies":
    st.markdown(
        '<div class="section-heading">Tools behind the magic</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="soft-card">
        <h3>🐍 Python</h3>
        The programming language used to implement the application.
    </div>
    <div class="soft-card">
        <h3>🧠 YOLO</h3>
        The pretrained AI model used to identify supported objects.
    </div>
    <div class="soft-card">
        <h3>🖥️ Streamlit</h3>
        Creates the interactive web interface.
    </div>
    <div class="soft-card">
        <h3>🖼️ OpenCV and Pillow</h3>
        Help convert and handle image data.
    </div>
    <div class="soft-card">
        <h3>📊 Pandas</h3>
        Organizes detection results into tables.
    </div>
    """, unsafe_allow_html=True)

    st.stop()


# ==================================================
# IMAGE UPLOAD AND SETTINGS
# ==================================================

st.markdown(
    '<div class="section-heading">📸 Let\'s detect something</div>',
    unsafe_allow_html=True,
)

upload_col, settings_col = st.columns([1.5, 1])

with upload_col:
    uploaded_file = st.file_uploader(
        "Choose your image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear photo in JPG, JPEG, or PNG format.",
    )

with settings_col:
    confidence = st.slider(
        "🎯 Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        help=(
            "Higher thresholds usually show fewer detections. "
            "They do not guarantee greater accuracy."
        ),
    )

    st.markdown("""
    <div class="soft-note">
        <b>Little tip 🌱</b><br>
        Clear, well-lit images often make objects easier to detect.
    </div>
    """, unsafe_allow_html=True)


# ==================================================
# DETECTION
# ==================================================

if uploaded_file is not None:
    try:
        original_image = Image.open(
            BytesIO(uploaded_file.getvalue())
        ).convert("RGB")

        st.markdown(
            '<div class="section-heading">🖼️ Your image</div>',
            unsafe_allow_html=True,
        )

        st.image(
            original_image,
            caption="Original upload",
            width="stretch",
        )

        detect_clicked = st.button(
            "🌿 Detect Objects",
            type="primary",
            use_container_width=True,
        )

        if detect_clicked:
            with st.spinner(
                "Your AI is looking at the image... ✨"
            ):
                @st.cache_resource
                def load_model():
                    return YOLO("yolo26n.pt")

                model = load_model()

                result = model.predict(
                    source=original_image,
                    conf=confidence,
                    verbose=False,
                )[0]

                annotated_bgr = result.plot()

                annotated_rgb = cv2.cvtColor(
                    annotated_bgr,
                    cv2.COLOR_BGR2RGB,
                )

                annotated_image = Image.fromarray(annotated_rgb)

                detections = []

                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        score = float(box.conf.item())

                        detections.append({
                            "Object": result.names[class_id],
                            "Confidence (%)": round(score * 100, 2),
                        })

            st.divider()

            st.markdown(
                '<div class="section-heading">✨ Your results</div>',
                unsafe_allow_html=True,
            )

            image_col, detected_col = st.columns(2)

            with image_col:
                st.markdown("**Original image**")
                st.image(
                    original_image,
                    width="stretch",
                )

            with detected_col:
                st.markdown("**AI-detected image**")
                st.image(
                    annotated_image,
                    width="stretch",
                )

            # SUMMARY METRICS

            total_detections = len(detections)

            unique_objects = len(
                set(item["Object"] for item in detections)
            )

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "📦 Total detections",
                total_detections,
            )

            metric2.metric(
                "🌱 Unique object classes",
                unique_objects,
            )

            metric3.metric(
                "🎯 Confidence setting",
                f"{confidence:.2f}",
            )

            if not detections:
                st.warning(
                    "No supported objects were detected. "
                    "Try a clearer image or a lower threshold."
                )

            else:
                # OBJECT COUNTS

                counts = Counter(
                    item["Object"] for item in detections
                )

                counts_df = pd.DataFrame([
                    {"Object": name, "Count": count}
                    for name, count in sorted(counts.items())
                ])

                st.markdown(
                    '<div class="section-heading">📊 Object counts</div>',
                    unsafe_allow_html=True,
                )

                st.dataframe(
                    counts_df,
                    hide_index=True,
                    use_container_width=True,
                )

                # INDIVIDUAL DETECTIONS

                st.markdown(
                    '<div class="section-heading">🔎 Detection details</div>',
                    unsafe_allow_html=True,
                )

                detection_df = pd.DataFrame(detections)

                st.dataframe(
                    detection_df,
                    hide_index=True,
                    use_container_width=True,
                )

            # DOWNLOAD ANNOTATED IMAGE

            image_buffer = BytesIO()
            annotated_image.save(image_buffer, format="PNG")

            st.download_button(
                label="⬇️ Download Detected Image (PNG)",
                data=image_buffer.getvalue(),
                file_name="sage_vision_detected.png",
                mime="image/png",
                use_container_width=True,
            )

    except UnidentifiedImageError:
        st.error(
            "This file could not be opened as an image. "
            "Please upload a valid JPG, JPEG, or PNG file."
        )

    except Exception as exc:
        st.error(
            "Something went wrong while processing the image."
        )
        st.exception(exc)


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.markdown("""
<div class="footer">
     AI-Based Image Object Detection System<br>
    Built with Python · Streamlit · YOLO · OpenCV · Pandas
</div>
""", unsafe_allow_html=True)
