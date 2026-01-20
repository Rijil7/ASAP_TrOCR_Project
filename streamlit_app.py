# streamlit_app.py
# VISIO Hybrid OCR – Full Document + Handwritten OCR
# Uses Tesseract for full documents (printed)
# Uses TrOCR for cropped / handwritten text
# Produces RAW output and ACCURATE (post-processed) output

import io
import re
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import cv2
import pytesseract
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from gtts import gTTS
import tempfile



# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="VISIO Hybrid OCR", layout="wide")
TARGET_SIZE = (1024, 1024)

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Handwritten
    proc_hw = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    model_hw = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-large-handwritten"
    ).to(device).eval()

    # Printed
    proc_pr = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
    model_pr = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-large-printed"
    ).to(device).eval()

    # Curved (fine-tuned TrOCR)
    proc_cv = TrOCRProcessor.from_pretrained("rijilraj77/trocr-curved-finetuned")
    model_cv = VisionEncoderDecoderModel.from_pretrained(
        "rijilraj77/trocr-curved-finetuned"
    ).to(device).eval()

    return {
        "handwritten": {
            "engine": "trocr",
            "processor": proc_hw,
            "model": model_hw
        },
        "printed": {
            "engine": "trocr",
            "processor": proc_pr,
            "model": model_pr
        },
        "curved": {
            "engine": "trocr",
            "processor": proc_cv,
            "model": model_cv
        },
        "device": device
    }

# -----------------------------
# PREPROCESSING
# -----------------------------
def preprocess_light(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    return img

# -----------------------------
# OCR ENGINES
# -----------------------------
def ocr_tesseract(img: Image.Image) -> str:
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return pytesseract.image_to_string(cv_img, lang="eng")


def ocr_trocr(img: Image.Image, proc, model) -> str:
    img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    px = proc(images=img, return_tensors="pt").pixel_values
    px = px.to(model.device)
    
    with torch.no_grad():
        ids = model.generate(px, max_length=256)

    return proc.batch_decode(ids, skip_special_tokens=True)[0]

def generate_speech(text):
    if not text.strip():
        return None
    tts = gTTS(text=text, lang="en")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    return tmp.name

# -----------------------------
# POST-PROCESSING
# -----------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.replace("|", "I")
    return text.strip()

# -----------------------------
# UI
# -----------------------------
st.title("VISIO Hybrid OCR System")
st.caption("Document OCR (Tesseract) + Handwritten OCR (TrOCR)")

# Session state
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""

if "clean_text" not in st.session_state:
    st.session_state.clean_text = ""

# LOAD MODELS ONCE HERE
models = load_models()

mode = st.radio(
    "Select OCR Mode",
    ["Printed", "Handwritten","Curved"]
)

uploaded = st.file_uploader(
    "Upload image",
    type=["png", "jpg", "jpeg"]
)

if uploaded:
    img = Image.open(uploaded)
    st.image(img, width=700)

    if st.button("Run OCR"):
        with st.spinner("Running OCR…"):
            img_prep = preprocess_light(img)
            raw = ""

            if mode == "Printed":
                 proc = models["printed"]["processor"]
                 model = models["printed"]["model"]
                 raw = ocr_trocr(img_prep, proc, model)

            elif mode == "Handwritten":
                proc = models["handwritten"]["processor"]
                model = models["handwritten"]["model"]
                raw = ocr_trocr(img_prep, proc, model)

            elif mode == "Curved":
                proc = models["curved"]["processor"]
                model = models["curved"]["model"]
                raw = ocr_trocr(img_prep, proc, model)

            st.session_state.raw_text = raw
            st.session_state.clean_text = clean_text(raw)

        st.success("OCR completed")

# Output section
if st.session_state.raw_text:
    st.subheader("Raw OCR Output")
    st.text_area("Raw", st.session_state.raw_text, height=200)

    st.subheader("Accurate Output (Post-processed)")
    st.text_area("Cleaned", st.session_state.clean_text, height=200)

    st.subheader("Audio Output (Text-to-Speech)")
    if st.checkbox("Generate Audio (TTS)"):
        audio_file = generate_speech(st.session_state.clean_text)
        st.audio(audio_file)


st.markdown("---")
st.markdown(
    "**Notes:**\n" \
    "• Upload clear images containing a single word for best results \n"
    "• Works for Printed, Handwritten, and Curved text \n"
    "• Crop the image tightly around the word for higher accuracy \n"
)
