import io
import os
import re
from typing import List

import fitz
import numpy as np
import pytesseract
import requests
import streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer


st.set_page_config(page_title="Phi-4 Document Reader", page_icon="📄", layout="wide")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
MODEL = os.getenv("OLLAMA_MODEL", "phi4-mini")


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ocr_image(image: Image.Image) -> str:
    return clean_text(pytesseract.image_to_string(image.convert("RGB")))


def extract_pdf(data: bytes) -> tuple[str, int]:
    document = fitz.open(stream=data, filetype="pdf")
    pages: List[str] = []
    ocr_pages = 0
    for number, page in enumerate(document, start=1):
        text = clean_text(page.get_text("text"))
        if len(text) < 40:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            text = ocr_image(image)
            ocr_pages += 1
        pages.append(f"--- Page {number} ---\n{text}")
    document.close()
    return "\n\n".join(pages), ocr_pages


def extract_document(uploaded_file) -> tuple[str, int]:
    data = uploaded_file.getvalue()
    if uploaded_file.name.lower().endswith(".pdf"):
        return extract_pdf(data)
    return ocr_image(Image.open(io.BytesIO(data))), 1


def make_chunks(text: str, size: int = 3500, overlap: int = 350) -> List[str]:
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            paragraph_end = text.rfind("\n", start, end)
            if paragraph_end > start + size // 2:
                end = paragraph_end
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def relevant_context(question: str, text: str, top_k: int = 5) -> str:
    chunks = make_chunks(text)
    if len(chunks) <= top_k:
        return "\n\n".join(chunks)
    matrix = TfidfVectorizer(stop_words="english").fit_transform([question] + chunks)
    scores = (matrix[1:] @ matrix[0].T).toarray().ravel()
    selected = np.argsort(scores)[::-1][:top_k]
    return "\n\n".join(chunks[index] for index in sorted(selected))


def ask_phi(system_prompt: str, user_prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": 0.2},
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def model_is_ready() -> bool:
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).ok
    except requests.RequestException:
        return False


st.title("📄 Phi-4 Document Reader")
st.caption("Private, local document OCR, summaries, and question answering with Phi-4 Mini")

with st.sidebar:
    st.subheader("Local model")
    if model_is_ready():
        st.success(f"Connected to {MODEL}")
    else:
        st.error("Ollama is not reachable")
        st.code("ollama serve\nollama pull phi4-mini", language="bash")
    st.info("Uploaded content is processed by this Streamlit server. With local Ollama, it is not sent to an AI cloud API.")

uploaded = st.file_uploader(
    "Upload a PDF or scanned document",
    type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"],
    max_upload_size=25,
)

if uploaded:
    file_key = f"{uploaded.name}:{uploaded.size}"
    if st.session_state.get("file_key") != file_key:
        with st.spinner("Reading document and running OCR where needed..."):
            try:
                text, ocr_pages = extract_document(uploaded)
                st.session_state.update(file_key=file_key, document_text=text, ocr_pages=ocr_pages)
            except Exception as exc:
                st.error(f"Could not read this file: {exc}")
                st.stop()

    text = st.session_state.document_text
    if not text.strip():
        st.warning("No readable text was found. Try a clearer scan or install the correct Tesseract language pack.")
        st.stop()

    words = len(text.split())
    col1, col2, col3 = st.columns(3)
    col1.metric("Words", f"{words:,}")
    col2.metric("Characters", f"{len(text):,}")
    col3.metric("OCR pages", st.session_state.ocr_pages)

    tab1, tab2, tab3 = st.tabs(["Extracted text", "AI summary", "Ask the document"])

    with tab1:
        st.text_area("Recognized text", text, height=480)
        st.download_button("Download text", text, file_name=f"{uploaded.name}.txt", mime="text/plain")

    with tab2:
        summary_style = st.selectbox("Summary type", ["Short overview", "Detailed summary", "Key facts and action items"])
        if st.button("Generate summary", type="primary"):
            if not model_is_ready():
                st.error("Start Ollama and pull phi4-mini first.")
            else:
                source = "\n\n".join(make_chunks(text)[:8])
                prompt = f"Create a {summary_style.lower()} of the document below. Use only the document. State when something is unclear.\n\nDOCUMENT:\n{source}"
                with st.spinner("Phi-4 is reading..."):
                    try:
                        st.session_state.summary = ask_phi("You are a careful document analyst.", prompt)
                    except requests.RequestException as exc:
                        st.error(f"Model request failed: {exc}")
        if st.session_state.get("summary"):
            st.markdown(st.session_state.summary)

    with tab3:
        question = st.text_input("Ask a question", placeholder="What are the main obligations in this document?")
        if st.button("Ask Phi-4") and question:
            if not model_is_ready():
                st.error("Start Ollama and pull phi4-mini first.")
            else:
                context = relevant_context(question, text)
                prompt = f"Answer the question using only the supplied document excerpts. If the answer is absent, say so.\n\nQUESTION:\n{question}\n\nDOCUMENT EXCERPTS:\n{context}"
                with st.spinner("Finding the answer..."):
                    try:
                        answer = ask_phi("You answer questions grounded strictly in the provided document.", prompt)
                        st.markdown(answer)
                    except requests.RequestException as exc:
                        st.error(f"Model request failed: {exc}")
else:
    st.info("Upload a document to begin. Digital PDFs use embedded text; scanned pages and images use OCR automatically.")

