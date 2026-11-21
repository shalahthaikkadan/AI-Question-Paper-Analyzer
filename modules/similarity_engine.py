
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Pretrained Sentence Transformer model
MODEL_NAME = "all-MiniLM-L6-v2"


def load_model():
    """Load pretrained Sentence Transformer model."""

    print("Loading AI semantic similarity model...")

    model = SentenceTransformer(MODEL_NAME)

    print("AI model loaded successfully!")

    return model


def calculate_similarity(df, model, threshold=0.85):
    """
    Find semantically similar questions across different PDFs.

    Returns:
        DataFrame containing similar question pairs,
        similarity scores, metadata and row indexes.
    """

    # Check whether the dataset is empty
    if df.empty:
        print("No questions available for similarity analysis.")
        return pd.DataFrame()

    # Create a copy and reset indexes
    df = df.reset_index(drop=True).copy()

    # Remove empty question text
    df = df[
        df["normalized_text"].notna()
        & (df["normalized_text"].str.strip() != "")
    ].copy()

    df.reset_index(drop=True, inplace=True)

    # Check whether enough questions are available
    if len(df) < 2:
        print("Not enough questions to compare.")
        return pd.DataFrame()

    print(f"Generating embeddings for {len(df)} questions...")

    # Generate numerical embeddings
    embeddings = model.encode(
        df["normalized_text"].tolist(),
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    repeated_questions = []

    print("Comparing question meanings...")

    # Compare each unique question pair
    for i in range(len(df)):

        for j in range(i + 1, len(df)):

            q1 = df.iloc[i]
            q2 = df.iloc[j]

            # Ignore questions from the same PDF
            if q1["source_file"] == q2["source_file"]:
                continue

            # Keep Part A and Part B separate
            if q1["part"] != q2["part"]:
                continue

            # Keep different known modules separate
            module1 = q1["module"]
            module2 = q2["module"]

            if pd.notna(module1) and pd.notna(module2):
                if module1 != module2:
                    continue

            # Get similarity score
            score = float(similarity_matrix[i][j])

            # Store similar question pairs
            if score >= threshold:

                repeated_questions.append({
                    # Row indexes used by the clustering module
                    "index_1": int(i),
                    "index_2": int(j),

                    # Similarity score
                    "similarity_score": round(score, 4),

                    # Question texts
                    "question_1": q1["text"],
                    "question_2": q2["text"],

                    # Exam years
                    "year_1": q1["year"],
                    "year_2": q2["year"],

                    # Question parts
                    "part_1": q1["part"],
                    "part_2": q2["part"],

                    # Modules
                    "module_1": module1,
                    "module_2": module2,

                    # Source PDF files
                    "source_1": q1["source_file"],
                    "source_2": q2["source_file"]
                })

    # Convert results into DataFrame
    result_df = pd.DataFrame(repeated_questions)

    # Sort by highest similarity score
    if not result_df.empty:

        result_df = result_df.sort_values(
            by="similarity_score",
            ascending=False
        ).reset_index(drop=True)

        print(
            f"\nSimilar question pairs found: {len(result_df)}"
        )

    else:
        print("\nNo similar questions found.")

    return result_df
