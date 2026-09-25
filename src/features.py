"""Feature extraction: skills, years of experience and education level."""
from __future__ import annotations

import datetime
import re

import pandas as pd

from .skills import ALIASES, ALL_SKILLS

EDUCATION_LABELS = {0: "Not stated", 1: "Diploma", 2: "Bachelor's", 3: "Master's", 4: "PhD"}


def _norm(text: str) -> str:
    return str(text).replace("\u2019", "'").replace("\u2018", "'")


# --------------------------------------------------------------------------- #
# Skills
# --------------------------------------------------------------------------- #
def _compile(term: str) -> re.Pattern:
    # Whole-term match; the lookarounds stop "sql" matching inside "mysql" and
    # let terms such as "c++", "c#" and "node.js" work.
    return re.compile(r"(?<![\w+#.])" + re.escape(term) + r"(?![\w+#])", re.IGNORECASE)


_SKILL_PATTERNS = [(s, _compile(s)) for s in ALL_SKILLS] + [
    (canonical, _compile(alias)) for alias, canonical in ALIASES.items()
]


def extract_skills(text: str) -> set[str]:
    """Return the set of canonical skills mentioned in `text`."""
    text = _norm(text)
    return {skill for skill, pattern in _SKILL_PATTERNS if pattern.search(text)}


# --------------------------------------------------------------------------- #
# Experience
# --------------------------------------------------------------------------- #
EXPLICIT_YEARS_RE = re.compile(
    r"(\d{1,2})\s*\+?\s*(?:-\s*\d{1,2}\s*)?(?:years?|yrs?)['\u2019]?\s+(?:of\s+)?"
    r"(?:[\w\-/&,]+\s+){0,4}?experience",
    re.IGNORECASE,
)
_MONTH = r"(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+|\d{1,2}[/\-])?"
_RANGE_RE = re.compile(
    rf"(?<!\d)((?:19|20)\d{{2}})\s*(?:-|\u2013|\u2014|to|till)\s*{_MONTH}((?:19|20)\d{{2}}|present|current|now|date)",
    re.IGNORECASE,
)
_EDU_CTX_RE = re.compile(
    r"\b(university|college|institute|school|academy|b\.?\s?tech|m\.?\s?tech|bachelor|master'?s|master\s+of|"
    r"b\.?\s?sc|m\.?\s?sc|mba|degree|diploma|cgpa|gpa|graduat\w*|ph\.?\s?d)\b",
    re.IGNORECASE,
)
_EDU_CTX_BE_RE = re.compile(r"\bB\.\s?E\b")  # case-sensitive so the word "be" doesn't match


def _is_education_context(text: str, m: re.Match) -> bool:
    line_start = text.rfind("\n", 0, m.start()) + 1
    line_end = text.find("\n", m.end())
    line_end = len(text) if line_end == -1 else line_end
    ctx = text[max(line_start, m.start() - 50): min(line_end, m.end() + 50)]
    return bool(_EDU_CTX_RE.search(ctx) or _EDU_CTX_BE_RE.search(ctx))


def extract_experience_years(text: str, current_year: int | None = None) -> float:
    """Estimate years of professional experience.

    1. Prefer an explicit statement such as "3+ years of experience".
    2. Otherwise merge the year ranges found in the text ("2019 - Present"),
       ignoring ranges that look like education entries.
    """
    cy = current_year or datetime.date.today().year
    text = _norm(text)

    explicit = [float(m.group(1)) for m in EXPLICIT_YEARS_RE.finditer(text) if float(m.group(1)) <= 45]
    if explicit:
        return max(explicit)

    spans = []
    for m in _RANGE_RE.finditer(text):
        if _is_education_context(text, m):
            continue
        start = int(m.group(1))
        end_raw = m.group(2).lower()
        end = int(end_raw) if end_raw.isdigit() else cy
        if start <= end <= cy and end - start <= 45:
            spans.append((start, end))

    spans.sort()
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return float(sum(e - s for s, e in merged))


# --------------------------------------------------------------------------- #
# Education
# --------------------------------------------------------------------------- #
_EDU_PATTERNS = [
    (4, re.compile(r"\b(?:ph\.?\s?d|doctorate|doctoral)\b", re.IGNORECASE)),
    (3, re.compile(
        r"\b(?:master(?:'s|s)?\s+(?:of|in|degree)|master'?s|m\.?\s?tech|m\.?\s?sc|mba|mca|pgdm|post[\s-]?graduat\w*)\b",
        re.IGNORECASE)),
    (2, re.compile(
        r"\b(?:bachelor(?:'s|s)?|b\.?\s?tech|b\.?\s?sc|b\.?\s?com|bca|bba|under[\s-]?graduat\w*)\b",
        re.IGNORECASE)),
    (2, re.compile(r"\bB\.\s?E\b")),  # case-sensitive: avoids the word "be"
    (1, re.compile(r"\bdiploma\b", re.IGNORECASE)),
]


def education_levels(text: str) -> set[int]:
    text = _norm(text)
    return {level for level, pattern in _EDU_PATTERNS if pattern.search(text)}


def extract_education_level(text: str) -> int:
    """Highest education level mentioned (0 = not stated ... 4 = PhD)."""
    levels = education_levels(text)
    return max(levels) if levels else 0


# --------------------------------------------------------------------------- #
# DataFrame helper
# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame, current_year: int | None = None) -> pd.DataFrame:
    """Add `skills`, `experience_years` and `education_level` columns to df (needs a `text` column)."""
    out = df.copy()
    out["skills"] = out["text"].apply(lambda t: sorted(extract_skills(t)))
    out["experience_years"] = out["text"].apply(lambda t: extract_experience_years(t, current_year))
    out["education_level"] = out["text"].apply(extract_education_level)
    return out
