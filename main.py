# ============================================================
# app.py — OCR Evaluation and Manual Approval Tool
# ============================================================

import os
import zipfile
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import easyocr
import pytesseract
from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import google.generativeai as genai

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ------------------- CONFIGURATION --------------------------
st.set_page_config(page_title="OCR Evaluation Dashboard", layout="wide")

st.title("🧠 OCR Benchmark & Manual Approval")
st.write("Test EasyOCR, Tesseract, TrOCR, PaddleOCR, and Gemini on your invoices dataset")

# (Optional) Windows path for tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ------------------- FILE UPLOAD -----------------------------
uploaded_zip = st.file_uploader("📂 Upload input_images.zip", type="zip")

if uploaded_zip:
    extract_dir = "images"
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    image_files = sorted([
        os.path.join(extract_dir, f)
        for f in os.listdir(extract_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    st.success(f"✅ {len(image_files)} images loaded")

    # ------------------- MODEL SETUP -------------------------
    st.sidebar.header("Model Setup")
    genai_api = st.sidebar.text_input("Gemini API Key (optional)", type="password")

    st.sidebar.write("Initializing OCR Models...")
    easy_reader = easyocr.Reader(['en'])
    paddle_reader = PaddleOCR(use_angle_cls=True, lang='en')
    trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
    trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

    if genai_api:
        genai.configure(api_key=genai_api)

    # ------------------- OCR FUNCTIONS -----------------------
    def ocr_easy(img_path):
        result = easy_reader.readtext(img_path, detail=1)
        text = " ".join([r[1] for r in result])
        boxes = [((r[0][0][0], r[0][0][1], r[0][2][0]-r[0][0][0], r[0][2][1]-r[0][0][1]), r[1]) for r in result]
        return text, boxes

    def ocr_tesseract(img_path):
        img = Image.open(img_path)
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        text = pytesseract.image_to_string(img)
        boxes = []
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:
                boxes.append(((data['left'][i], data['top'][i], data['width'][i], data['height'][i]), data['text'][i]))
        return text, boxes

    def ocr_paddle(img_path):
        result = paddle_reader.ocr(img_path, cls=True)
        boxes = []
        text = ""
        for line in result[0]:
            text += line[1][0] + " "
            box = line[0]
            boxes.append(((box[0][0], box[0][1], box[2][0]-box[0][0], box[2][1]-box[0][1]), line[1][0]))
        return text, boxes

    def ocr_trocr(img_path):
        image = Image.open(img_path).convert("RGB")
        pixel_values = trocr_processor(images=image, return_tensors="pt").pixel_values
        generated_ids = trocr_model.generate(pixel_values)
        text = trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text, []

    def ocr_gemini(img_path):
        if not genai_api:
            return "(Gemini not configured)", []
        image = Image.open(img_path)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(["Extract text from this invoice image.", image])
        return response.text, []

    models = {
        "Tesseract": ocr_tesseract,
        "EasyOCR": ocr_easy,
        "PaddleOCR": ocr_paddle,
        "TrOCR": ocr_trocr,
        "Gemini": ocr_gemini
    }

    # ------------------- MAIN LOOP ---------------------------
    results = []

    for img_path in image_files:
        st.subheader(f"🧾 {os.path.basename(img_path)}")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(img_path, caption="Original Image", use_container_width=True)

        with col2:
            for model_name, func in models.items():
                st.markdown(f"### {model_name}")
                text, boxes = func(img_path)

                # Draw boxes
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                for (x, y, w, h), _ in boxes:
                    cv2.rectangle(img_rgb, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 2)

                annotated_path = f"annotated_{model_name}_{os.path.basename(img_path)}"
                cv2.imwrite(annotated_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

                st.image(img_rgb, caption=f"{model_name} Text Overlay", use_container_width=True)
                st.text_area(f"Detected Text ({model_name})", text, height=100)
                approve = st.radio(f"Approve {model_name} result?", ["Pending", "✅ Yes", "❌ No"], key=f"{img_path}_{model_name}")

                results.append({
                    "Image": os.path.basename(img_path),
                    "Model": model_name,
                    "Detected_Text": text,
                    "Approved": approve
                })

                st.download_button(
                    f"📥 Download {model_name} annotated image",
                    data=open(annotated_path, "rb"),
                    file_name=annotated_path
                )

    # ------------------- SAVE RESULTS ------------------------
    if st.button("💾 Save All Approvals"):
        df = pd.DataFrame(results)
        df.to_csv("manual_approval_results.csv", index=False)
        st.success("✅ Saved manual_approval_results.csv")
        st.download_button(
            "⬇️ Download Results CSV",
            data=open("manual_approval_results.csv", "rb"),
            file_name="manual_approval_results.csv"
        )
