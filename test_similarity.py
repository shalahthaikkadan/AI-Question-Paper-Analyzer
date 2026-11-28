
from pathlib import Path
import pandas as pd

from modules.similarity_engine import (
    load_model,
    calculate_similarity
)


# Project directories
BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "output" / "prepared_questions.csv"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "similar_questions.csv"


# Check input file
if not INPUT_FILE.exists():
    print("Prepared questions file not found!")
    print("Run: python test_parser.py")
    raise SystemExit(1)


# Load prepared questions
print("Loading prepared questions...")

df = pd.read_csv(INPUT_FILE)

print("Total questions:", len(df))


# Check required columns
required_columns = [
    "normalized_text",
    "source_file",
    "part",
    "module",
    "text",
    "year"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print("Missing columns:", missing_columns)
    raise SystemExit(1)


# Load AI model
model = load_model()


# Calculate semantic similarity
similarity_df = calculate_similarity(
    df,
    model,
    threshold=0.85
)


# Save similarity results
if similarity_df.empty:
    print("No similar questions found.")
    
    # Save an empty CSV with the expected columns
    similarity_df = pd.DataFrame(columns=[
        "index_1",
        "index_2",
        "similarity_score",
        "question_1",
        "question_2",
        "year_1",
        "year_2",
        "part_1",
        "part_2",
        "module_1",
        "module_2",
        "source_1",
        "source_2"
    ])

similarity_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nSimilarity analysis completed!")
print("Similar question pairs:", len(similarity_df))
print("Results saved at:", OUTPUT_FILE)
