
import re
import pandas as pd


# -------------------------------------------------
# 1. Extract exam year
# -------------------------------------------------

def extract_year(filename, pdf_text=""):
    """
    Extract exam year from PDF filename or PDF content.
    """

    # First check filename
    match = re.search(
        r"\b(20\d{2})\b",
        str(filename)
    )

    if match:
        return int(match.group(1))

    # Check PDF text for exam year
    patterns = [
        r"examination\D{0,60}(20\d{2})",
        r"exam\D{0,40}(20\d{2})",
        r"\b(?:January|February|March|April|May|June|July|"
        r"August|September|October|November|December)"
        r"\s+(20\d{2})\b"
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            str(pdf_text),
            flags=re.IGNORECASE
        )

        if match:
            return int(match.group(1))

    return None


# -------------------------------------------------
# 2. Extract subject code
# -------------------------------------------------

def extract_subject(filename):
    """
    Extract subject code such as CST423.
    """

    match = re.search(
        r"\b([A-Z]{2,4}\d{3})\b",
        str(filename).upper()
    )

    if match:
        return match.group(1)

    return "UNKNOWN"


# -------------------------------------------------
# 3. Extract question marks
# -------------------------------------------------

def extract_marks(question_text):
    """
    Extract marks from the end of a question.
    Example: Explain cloud computing. (8)
    """

    match = re.search(
        r"\((\d+)\)\s*$",
        str(question_text)
    )

    if match:
        return int(match.group(1))

    return None


# -------------------------------------------------
# 4. Normalize question text
# -------------------------------------------------

def normalize_question(question_text):
    """
    Clean question text for AI semantic comparison.
    """

    text = str(question_text).lower()

    # Remove marks at the end, e.g. (8)
    text = re.sub(
        r"\(\d+\)\s*$",
        "",
        text
    )

    # Remove punctuation
    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# -------------------------------------------------
# 5. Prepare complete dataset
# -------------------------------------------------

def prepare_dataset(df):
    """
    Prepare extracted questions for AI analysis.
    """

    if df.empty:
        raise ValueError(
            "No questions available for preprocessing."
        )

    df = df.copy()

    # Extract year
    if "exam_year" in df.columns:

        filename_years = df["source_file"].apply(
            lambda name: extract_year(name)
        )

        df["year"] = df["exam_year"].fillna(
            filename_years
        )

    else:

        df["year"] = df["source_file"].apply(
            lambda name: extract_year(name)
        )

    # Extract subject
    df["subject"] = df["source_file"].apply(
        extract_subject
    )

    # Extract marks
    df["marks"] = df["text"].apply(
        extract_marks
    )

    # Normalize question text
    df["normalized_text"] = df["text"].apply(
        normalize_question
    )

    # Remove empty questions
    df = df[
        df["normalized_text"].notna()
        & (df["normalized_text"].str.strip() != "")
    ].copy()

    # Reset index
    df.reset_index(
        drop=True,
        inplace=True
    )

    return df
