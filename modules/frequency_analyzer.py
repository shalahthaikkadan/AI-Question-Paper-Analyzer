
import pandas as pd


def analyze_question_frequency(cluster_df):
    """
    Analyze repeated question frequency.

    Calculates:
    - Total occurrences
    - Number of unique question papers
    - Number of unique years
    - Years in which questions appeared
    """

    if cluster_df.empty:
        print("No question clusters available.")
        return pd.DataFrame()

    required_columns = [
        "cluster_id",
        "question",
        "part",
        "module",
        "year",
        "source_file"
    ]

    missing = [
        col for col in required_columns
        if col not in cluster_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = cluster_df.copy()

    # Group questions by cluster
    results = []

    for cluster_id, group in df.groupby("cluster_id"):

        # Remove missing years
        valid_years = group["year"].dropna()

        # Count unique papers
        total_papers = group["source_file"].nunique()

        # Count unique exam years
        total_years = valid_years.nunique()

        # Get years as a readable string
        years = sorted(
            valid_years.astype(int).unique().tolist()
        )

        # Select the longest question as representative
        representative_question = (
            group["question"]
            .dropna()
            .astype(str)
            .sort_values(key=lambda x: x.str.len(), ascending=False)
            .iloc[0]
        )

        results.append({
            "cluster_id": cluster_id,
            "question": representative_question,
            "part": group["part"].iloc[0],
            "module": group["module"].iloc[0],
            "total_occurrences": len(group),
            "total_papers": total_papers,
            "total_years": total_years,
            "years_appeared": ", ".join(map(str, years))
        })

    frequency_df = pd.DataFrame(results)

    # Sort by number of papers, then occurrences
    frequency_df = frequency_df.sort_values(
        by=["total_papers", "total_occurrences"],
        ascending=False
    ).reset_index(drop=True)

    return frequency_df
