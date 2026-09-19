import io
import os

import fitz  # PyMuPDF
import pytesseract
import streamlit as st
from docx import Document
from openai import OpenAI
from PIL import Image

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MAX_CONTEXT_CHARS = 18000

st.set_page_config(page_title="Qwen Document Reader", page_icon="📄", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 2rem;}
[data-testid="stFileUploader"] {border: 1px dashed #7c8cff; border-radius: 14px; padding: 1rem;}
.hero {padding: 1.5rem; border-radius: 18px; background: linear-gradient(135deg,#1d2340,#303a73); color:white; margin-bottom:1rem;}
.small {opacity:.8; font-size:.92rem}
</style>
<div class="hero">
  <h1>📄 Qwen Document Reader</h1>
  <p>Upload a document, extract its text (including OCR), then summarize it or ask questions.</p>
</div>
""", unsafe_allow_html=True)


def get_token():
    try:
        return st.secrets["HF_TOKEN"]
    except (KeyError, FileNotFoundError):
        return os.getenv("HF_TOKEN", "")


@st.cache_data(show_spinner=False)
def extract_pdf(data: bytes):
    doc = fitz.open(stream=data, filetype="pdf")
    pages, ocr_pages = [], 0
    for number, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if len(text) < 30:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(image).strip()
            ocr_pages += 1
        pages.append(f"--- Page {number} ---\n{text}")
    return "\n\n".join(pages), ocr_pages, len(doc)


@st.cache_data(show_spinner=False)
def extract_image(data: bytes):
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return pytesseract.image_to_string(image).strip()


@st.cache_data(show_spinner=False)
def extract_docx(data: bytes):
    doc = Document(io.BytesIO(data))
    blocks = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            blocks.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(blocks)


def extract(upload):
    data = upload.getvalue()
    name = upload.name.lower()
    if name.endswith(".pdf"):
        text, ocr_pages, total_pages = extract_pdf(data)
        return text, f"{total_pages} page(s); OCR used on {ocr_pages}"
    if name.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
        return extract_image(data), "Image processed with OCR"
    if name.endswith(".docx"):
        return extract_docx(data), "Word document parsed"
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="replace"), "Text file read"
    raise ValueError("Unsupported file type")


def ask_model(instruction: str, document_text: str, token: str):
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=token,
    )
    clipped = document_text[:MAX_CONTEXT_CHARS]
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful document assistant. Answer only from the supplied "
                    "document. If the answer is absent, say that it was not found."
                ),
            },
            {
                "role": "user",
                "content": f"DOCUMENT:\n{clipped}\n\nTASK:\n{instruction}",
            },
        ],
        temperature=0.2,
        max_tokens=700,
    )
    return response.choices[0].message.content


with st.sidebar:
    st.header("Settings")
    st.caption(f"Model: {MODEL}")
    token = get_token()
    if token:
        st.success("HF token detected")
    else:
        st.warning("Add HF_TOKEN in Streamlit secrets")
    st.markdown(
        "Supported: **PDF, scanned PDF, PNG, JPG, TIFF, BMP, DOCX, TXT**"
    )
    st.caption("Files are processed during the session and are not intentionally stored.")

upload = st.file_uploader(
    "Upload a document",
    type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "docx", "txt"],
)

if upload:
    try:
        with st.spinner("Reading document and running OCR where needed..."):
            text, details = extract(upload)
    except Exception as exc:
        st.error(f"Could not read this document: {exc}")
        st.stop()

    if not text.strip():
        st.error("No readable text was found. Try a clearer scan.")
        st.stop()

    st.success(f"Text extracted — {details}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Characters", f"{len(text):,}")
    col2.metric("Words", f"{len(text.split()):,}")
    col3.download_button(
        "Download text",
        text,
        file_name=f"{upload.name.rsplit('.', 1)[0]}_text.txt",
        mime="text/plain",
        use_container_width=True,
    )

    tab_chat, tab_text = st.tabs(["Ask Qwen", "Extracted text"])

    with tab_chat:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        action_col1, action_col2, action_col3 = st.columns(3)
        summarize = action_col1.button("Summarize", use_container_width=True)
        key_points = action_col2.button("Key points", use_container_width=True)
        clear = action_col3.button("Clear chat", use_container_width=True)

        if clear:
            st.session_state.messages = []
            st.rerun()

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input("Ask a question about this document")
        instruction = (
            "Give a concise summary with headings."
            if summarize
            else "List the most important facts and action items."
            if key_points
            else prompt
        )

        if instruction:
            if not token:
                st.error(
                    "Add your Hugging Face access token as HF_TOKEN in "
                    "Streamlit Cloud → App settings → Secrets."
                )
            else:
                st.session_state.messages.append({"role": "user", "content": instruction})
                with st.chat_message("user"):
                    st.markdown(instruction)
                with st.chat_message("assistant"):
                    with st.spinner("Qwen is reading..."):
                        try:
                            answer = ask_model(instruction, text, token)
                            st.markdown(answer)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": answer}
                            )
                        except Exception as exc:
                            st.error(
                                "Model request failed. Confirm HF_TOKEN has Inference "
                                f"Providers permission. Details: {exc}"
                            )

    with tab_text:
        st.text_area("OCR / extracted text", text, height=520)
else:
    st.info("Upload a file to begin. A sample PDF or phone photo works well.")
