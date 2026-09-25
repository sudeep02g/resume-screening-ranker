"""Reading resumes (CSV / PDF / DOCX / TXT), cleaning text and redacting personal details."""
from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

TEXT_COLUMNS = ["Resume_str", "resume_str", "resume_text", "Resume", "resume", "text", "Text"]
ID_COLUMNS = ["ID", "id", "Id", "resume_id"]
CATEGORY_COLUMNS = ["Category", "category"]

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+")
_PHONE_RE = re.compile(r"(?<![\w/])\+?\(?\d[\d\s().-]{8,}\d(?![\w/])")


def clean_text(text: str) -> str:
    """Normalise quotes/whitespace while keeping line breaks (used by the experience parser)."""
    text = str(text).replace("\u2019", "'").replace("\u2018", "'").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def redact_pii(text: str) -> str:
    """Remove emails, URLs and phone numbers so they can never influence scoring."""
    text = _EMAIL_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    # Only treat a digit run as a phone number if it has 10-15 digits, so year ranges like "2019 - 2022" survive.
    text = _PHONE_RE.sub(lambda m: " " if 10 <= sum(c.isdigit() for c in m.group()) <= 15 else m.group(), text)
    return text


# --------------------------------------------------------------------------- #
# File readers
# --------------------------------------------------------------------------- #
def extract_text_from_bytes(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        import pdfplumber

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "\n".join((page.extract_text() or "") for page in pdf.pages)
    if suffix == ".docx":
        import docx

        document = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in document.paragraphs)
    return data.decode("utf-8", errors="ignore")


def read_file(path: str | Path) -> str:
    path = Path(path)
    return extract_text_from_bytes(path.name, path.read_bytes())


def _first_existing(columns, candidates):
    return next((c for c in candidates if c in columns), None)


def load_resumes(source: str | Path, redact: bool = True) -> pd.DataFrame:
    """Load resumes from a CSV file or a folder of PDF/DOCX/TXT files.

    Returns a DataFrame with columns: id, category (may be empty), text.
    CSV column names are auto-detected (works with the Kaggle "Resume Dataset").
    """
    source = Path(source)
    if source.is_dir():
        rows = []
        for f in sorted(source.iterdir()):
            if f.suffix.lower() in {".pdf", ".docx", ".txt"}:
                rows.append({"id": f.stem, "category": "", "text": read_file(f)})
        df = pd.DataFrame(rows, columns=["id", "category", "text"])
    else:
        raw = pd.read_csv(source)
        text_col = _first_existing(raw.columns, TEXT_COLUMNS)
        if text_col is None:
            raise ValueError(f"No resume text column found. Expected one of {TEXT_COLUMNS}, got {list(raw.columns)}")
        id_col = _first_existing(raw.columns, ID_COLUMNS)
        cat_col = _first_existing(raw.columns, CATEGORY_COLUMNS)
        df = pd.DataFrame({
            "id": raw[id_col] if id_col else range(len(raw)),
            "category": raw[cat_col] if cat_col else "",
            "text": raw[text_col],
        })

    df["text"] = df["text"].fillna("").map(clean_text)
    if redact:
        df["text"] = df["text"].map(redact_pii)
    df["id"] = df["id"].astype(str)
    return df.reset_index(drop=True)
