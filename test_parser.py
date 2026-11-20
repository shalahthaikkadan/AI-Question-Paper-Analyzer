
from pathlib import Path
import pymupdf
import pandas as pd

from modules.question_classifier import extract_questions


# 1. Project directories
BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Find PDF files
pdf_files = sorted(UPLOAD_DIR.glob("*.pdf"))

if not pdf_files:
    print("No PDF files found!")
    print("Place PDFs inside data/uploads/")
    raise SystemExit


# 3. Process all PDFs
all_questions = []

for pdf_path in pdf_files:

    print(f"\nProcessing: {pdf_path.name}")

    try:
        doc = pymupdf.open(str(pdf_path))

        full_text = ""

        for page in doc:
            full_text += page.get_text("text") + "\n"

        doc.close()

        # 4. Extract questions
        questions = extract_questions(full_text)

        # Add source filename to every record
        for question in questions:
            question["source_file"] = pdf_path.name

        all_questions.extend(questions)

        print(f"Questions extracted: {len(questions)}")

    except Exception as error:
        print(f"Error processing {pdf_path.name}: {error}")


# 5. Save results
df = pd.DataFrame(all_questions)

print("\n================================")
print("TOTAL QUESTIONS:", len(df))
print("================================")

if not df.empty:

    print(df.to_string(index=False))

    output_file = OUTPUT_DIR / "extracted_questions.csv"

    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("\nCSV generated successfully!")
    print("Saved at:", output_file)

else:
    print("No questions extracted.")
