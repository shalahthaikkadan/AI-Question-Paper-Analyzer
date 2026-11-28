
from pathlib import Path
import pymupdf
import pandas as pd

from modules.question_classifier import extract_questions
from modules.data_preprocessor import prepare_dataset, extract_year


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

        # 5. Detect exam year from filename or PDF text
        exam_year = extract_year(
            pdf_path.name,
            full_text
        )

        print("Detected exam year:", exam_year)

        # 6. Add metadata to every question
        for question in questions:
            question["source_file"] = pdf_path.name
            question["exam_year"] = exam_year

        all_questions.extend(questions)

        print(f"Questions extracted: {len(questions)}")

    except Exception as error:
        print(f"Error processing {pdf_path.name}: {error}")


# 7. Convert extracted questions into DataFrame
df = pd.DataFrame(all_questions)

print("\n================================")
print("TOTAL QUESTIONS:", len(df))
print("================================")


# 8. Validate extraction and prepare dataset
if not df.empty:

    print("\n========== EXTRACTION VALIDATION ==========")

    for filename in df["source_file"].unique():

        file_df = df[df["source_file"] == filename]

        part_a_count = len(
            file_df[file_df["part"] == "A"]
        )

        part_b_count = len(
            file_df[file_df["part"] == "B"]
        )

        print(f"\nPDF: {filename}")
        print(f"Part A: {part_a_count}")
        print(f"Part B: {part_b_count}")
        print(f"Total: {len(file_df)}")

        if part_a_count == 10 and part_b_count == 20:
            print("Status: PASS")
        else:
            print("Status: CHECK EXTRACTION")

    print("\n===========================================")

    # 9. Prepare dataset for AI analysis
    prepared_df = prepare_dataset(df)

    print("\n========== PREPROCESSING SUMMARY ==========")

    print("Total questions:", len(prepared_df))

    print("\nQuestions by Part:")
    print(prepared_df["part"].value_counts())

    print("\nQuestions by Year:")
    print(prepared_df["year"].value_counts(dropna=False))

    print("\nQuestions by Module:")
    print(prepared_df["module"].value_counts(dropna=False))

    # 10. Save extracted questions
    extracted_file = OUTPUT_DIR / "extracted_questions.csv"

    df.to_csv(
        extracted_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nExtracted CSV saved at:")
    print(extracted_file)

    # 11. Save prepared questions
    prepared_file = OUTPUT_DIR / "prepared_questions.csv"

    prepared_df.to_csv(
        prepared_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nPrepared CSV generated successfully!")
    print("Saved at:", prepared_file)

else:
    print("No questions extracted. Please check your PDF files.")
