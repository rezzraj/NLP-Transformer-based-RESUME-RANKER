import streamlit as st
import pandas as pd
import numpy as np
import re
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
# Avoid image size errors
Image.MAX_IMAGE_PIXELS = None

# ---------------------------
# Helper Functions
# ---------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer('all-mpnet-base-v2')


def ocr_extract_text(uploaded_file):
    # Save to a temporary file first
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    pages = convert_from_path(tmp_path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page) + "\n"
    return text.strip()


def remove_special_chars(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    text = re.sub(r'[^a-zA-Z0-9 .@\-/]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    text = " ".join(text.split())
    return text.strip()

def rank_resumes(job_description, resumes):
    model = load_model()

    # Clean job description
    job_description_clean = remove_special_chars(job_description)

    # Extract & clean resumes
    resume_data = []
    for resume_file in resumes:
        text = ocr_extract_text(resume_file)
        clean_text = remove_special_chars(text)
        resume_data.append({
            "filename": resume_file.name,
            "text": clean_text
        })

    df = pd.DataFrame(resume_data)

    # Encode embeddings
    resume_embeddings = model.encode(df['text'].tolist(), batch_size=16, show_progress_bar=False)
    job_embedding = model.encode([job_description_clean])

    # Cosine similarity
    scores = cosine_similarity(job_embedding, resume_embeddings)[0]
    df['score'] = scores

    # Sort by score
    df = df.sort_values(by='score', ascending=False).reset_index(drop=True)
    df.index = df.index + 1

    return df[['filename', 'score']]

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="AI Resume Ranker", layout="wide")
st.title("🚀 AI Resume Ranker")
st.write("Upload resumes and a job description, and let AI rank the best fit!")

# Job description input
job_desc_input = st.text_area("Paste Job Description Here:")

# Resume upload
resume_files = st.file_uploader("Upload Resume PDFs", type=["pdf"], accept_multiple_files=True)

# Rank button
if st.button("Rank Resumes"):
    if not job_desc_input:
        st.warning("Please paste a job description.")
    elif not resume_files:
        st.warning("Please upload at least one resume PDF.")
    else:
        with st.spinner("Processing resumes..."):
            results_df = rank_resumes(job_desc_input, resume_files)
        st.success("Ranking complete!")
        st.dataframe(results_df)

        # Download option
        csv_data = results_df.to_csv(index=True)
        st.download_button(
            label="Download Rankings as CSV",
            data=csv_data,
            file_name="resume_rankings.csv",
            mime="text/csv"
        )
