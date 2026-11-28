
from pathlib import Path
import pandas as pd

from modules.question_cluster import create_question_clusters


BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "output"

questions_file = OUTPUT_DIR / "prepared_questions.csv"
similarity_file = OUTPUT_DIR / "similar_questions.csv"


# Load datasets
questions_df = pd.read_csv(questions_file)

similarity_df = pd.read_csv(similarity_file)


# Create clusters
cluster_df = create_question_clusters(
    questions_df,
    similarity_df
)


print("\n========== QUESTION CLUSTERS ==========")

if not cluster_df.empty:

    for cluster_id in sorted(
        cluster_df["cluster_id"].unique()
    ):

        group = cluster_df[
            cluster_df["cluster_id"] == cluster_id
        ]

        print(f"\nCLUSTER {cluster_id}")

        for _, row in group.iterrows():

            print(
                f"- [{row['year']}] "
                f"{row['question']}"
            )

    # Save results
    output_file = OUTPUT_DIR / "question_clusters.csv"

    cluster_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nClusters saved:", output_file)

else:
    print("No repeated-question clusters found.")
