
from pathlib import Path
import pandas as pd

from modules.frequency_analyzer import analyze_question_frequency


BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "output"

CLUSTER_FILE = OUTPUT_DIR / "question_clusters.csv"

FREQUENCY_FILE = OUTPUT_DIR / "question_frequency.csv"


# Check cluster file
if not CLUSTER_FILE.exists():
    print("Question clusters file not found!")
    print("Run: python test_cluster.py")
    raise SystemExit(1)


# Load clusters
print("Loading question clusters...")

cluster_df = pd.read_csv(CLUSTER_FILE)

print("Total clustered questions:", len(cluster_df))


# Analyze frequency
print("\nAnalyzing question frequency...")

frequency_df = analyze_question_frequency(cluster_df)


# Display results
print("\n========== QUESTION FREQUENCY ==========")

if not frequency_df.empty:

    print(
        frequency_df.to_string(index=False)
    )

    frequency_df.to_csv(
        FREQUENCY_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nFrequency analysis completed!")
    print("Results saved at:", FREQUENCY_FILE)

else:
    print("No repeated questions found.")
