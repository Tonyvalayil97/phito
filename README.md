# Phi-4 Document Reader

A Streamlit demo that reads documents, performs OCR on scans, and uses Microsoft's **Phi-4 Mini Instruct** model to summarize the content or answer questions.

## Features

- Reads PDFs, scanned PDFs, images, DOCX and TXT
- Uses Tesseract OCR for scanned pages and photos
- Shows and downloads extracted text
- Summarizes documents and extracts key points
- Question-answering grounded in the uploaded document
- Designed for Streamlit Community Cloud

## Deploy on Streamlit Community Cloud

1. Open [share.streamlit.io](https://share.streamlit.io).
2. Click **Create app** and select this repository.
3. Set the main file path to `app.py`.
4. In **Advanced settings → Secrets**, add:

```toml
HF_TOKEN = "hf_your_token_here"
```

5. Deploy the app.

Create a token at [Hugging Face settings](https://huggingface.co/settings/tokens). The token needs permission to call Inference Providers. Never place the token directly in `app.py` or commit it to GitHub.

## Run locally

```bash
git clone https://github.com/Tonyvalayil97/phito.git
cd phito
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

For local OCR, install Tesseract separately:

- Windows: install Tesseract OCR and add it to PATH
- Ubuntu/Debian: `sudo apt install tesseract-ocr`
- macOS: `brew install tesseract`

Streamlit Cloud installs it automatically from `packages.txt`.

## How it works

1. PyMuPDF extracts text from normal PDFs.
2. Pages with little or no embedded text are rendered as images.
3. Tesseract OCR reads scanned pages and uploaded photos.
4. The extracted text is sent with the user's task to `microsoft/Phi-4-mini-instruct` through Hugging Face's OpenAI-compatible inference endpoint.

The demo limits model context to the first 18,000 characters to keep requests fast and reliable. For production, add chunking, retrieval, authentication, virus scanning, and durable storage.

## Troubleshooting

### `ModuleNotFoundError: No module named 'fitz'`

The import is `fitz`, but the package name is **PyMuPDF**. Confirm the file is named exactly `requirements.txt` and contains `PyMuPDF`.

### Tesseract is not installed

Confirm the repository contains a file named exactly `packages.txt` with:

```
tesseract-ocr
```

### Phi-4 request fails

Check that `HF_TOKEN` is present in Streamlit secrets and has permission to use Hugging Face Inference Providers.
