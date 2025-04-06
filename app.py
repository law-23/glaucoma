import streamlit as st
import SimpleITK as sitk
import numpy as np
import cv2
from PIL import Image
from fpdf import FPDF
import base64
import os

# ====================
# Helper Functions
# ====================

def preprocess_oct(file):
    try:
        temp_path = "temp_oct.mha"
        with open(temp_path, "wb") as f:
            f.write(file.read())
        itk_image = sitk.ReadImage(temp_path)
        image = sitk.GetArrayFromImage(itk_image)[0]
        image_resized = cv2.resize(image, (224, 224))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
        image_input = image_rgb / 255.0
        return np.expand_dims(image_input, axis=0), image_rgb
    except Exception as e:
        st.error(f"OCT preprocessing error: {e}")
        return None, None


def preprocess_fundus(file):
    try:
        img = Image.open(file).convert("RGB").resize((224, 224))
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0), img
    except Exception as e:
        st.error(f"Fundus preprocessing error: {e}")
        return None, None


def generate_pdf_report(label, source):
    font_path = "DejaVuSans.ttf"  # Make sure this font file is in the same directory
    pdf = FPDF()
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.add_font("DejaVu", "B", font_path, uni=True)

    pdf.add_page()
    pdf.set_font("DejaVu", "B", 16)
    pdf.cell(0, 10, "Glaucoma Screening Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("DejaVu", "", 12)

    pdf.cell(0, 10, f"Image Source: {source}", ln=True)
    pdf.cell(0, 10, f"Prediction: {label}", ln=True)
    pdf.ln(10)

    avail_w = pdf.w - pdf.l_margin - pdf.r_margin

    if label == "Normal":
        pdf.set_text_color(0, 128, 0)
        pdf.multi_cell(avail_w, 10, "No signs of Glaucoma detected.")
    else:
        pdf.set_text_color(255, 0, 0)
        pdf.multi_cell(avail_w, 10, "Signs of Glaucoma detected. Please consult a doctor.")
        pdf.ln(5)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("DejaVu", "", 12)
        pdf.cell(0, 10, "Precautions before your appointment:", ln=True)
        pdf.ln(2)
        precautions = [
            "Avoid straining your eyes; rest in a dimly lit room.",
            "Avoid heavy lifting or bending down.",
            "Limit caffeine intake before your visit.",
            "Note any recent changes in your vision and bring them up.",
            "Prepare a list of current medications and eye drops."
        ]
        pdf.set_font("DejaVu", "", 10)
        for idx, p in enumerate(precautions, 1):
            pdf.cell(0, 8, f"{idx}. {p}", ln=True)
        pdf.set_font("DejaVu", "", 12)

    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.multi_cell(avail_w, 10, "Note: This is an AI-assisted prediction. Always consult a specialist for confirmation.")

    fname = source.replace(" ", "_").replace("&", "and").lower()
    output_path = f"glaucoma_report_{fname}.pdf"
    pdf.output(output_path)
    return output_path

# ====================
# Streamlit App Layout
# ====================
st.set_page_config(layout="wide", page_title="Glaucoma Detection: OCT & Fundus")
st.title("Glaucoma Detection from OCT and Fundus Images")

col1, col2 = st.columns(2)

oct_label = fundus_label = None
oct_display = fundus_display = None

with col1:
    st.header("OCT Image (.mha)")
    oct_file = st.file_uploader("Upload OCT image", type=["mha"], key="oct_upload")
    if oct_file:
        with st.spinner("Processing OCT..."):
            oct_input, oct_display = preprocess_oct(oct_file)
        if oct_input is not None:
            oct_label = "Glaucoma"  # Hardcoded prediction
            st.image(oct_display, caption="OCT Scan", use_column_width=True)
            st.success(f"OCT Prediction: {oct_label}")

with col2:
    st.header("Fundus Image (jpg, png)")
    fundus_file = st.file_uploader("Upload Fundus image", type=["jpg", "jpeg", "png"], key="fundus_upload")
    if fundus_file:
        with st.spinner("Processing Fundus..."):
            fundus_input, fundus_display = preprocess_fundus(fundus_file)
        if fundus_input is not None:
            fundus_label = "Glaucoma"  # Hardcoded prediction
            st.image(fundus_display, caption="Fundus Image", use_column_width=True)
            st.success(f"Fundus Prediction: {fundus_label}")

st.markdown("---")
st.header("Report Generator")

if st.button("📄 Generate Report"):
    path = None
    if oct_label and not fundus_label:
        path = generate_pdf_report(oct_label, source="OCT")
    elif fundus_label and not oct_label:
        path = generate_pdf_report(fundus_label, source="Fundus")
    elif fundus_label and oct_label:
        if fundus_label == oct_label:
            path = generate_pdf_report(oct_label, source="OCT & Fundus")
        else:
            st.warning("OCT and Fundus predictions do not match. Please consult a medical professional.")
    else:
        st.error("No valid image uploaded.")

    if path:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        link = f'<a href="data:application/pdf;base64,{b64}" download="{os.path.basename(path)}">Download PDF Report</a>'
        st.markdown(link, unsafe_allow_html=True)
