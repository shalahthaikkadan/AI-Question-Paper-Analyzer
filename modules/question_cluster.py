
import pandas as pd
import networkx as nx


def create_question_clusters(questions_df, similarity_df):
    """
    Group semantically similar questions into clusters
    using similarity scores from the AI engine.
    """

    if questions_df.empty:
        print("No questions available.")
        return pd.DataFrame()

    if similarity_df.empty:
        print("No similar question pairs available.")
        return pd.DataFrame()

    questions_df = questions_df.reset_index(drop=True).copy()

    graph = nx.Graph()

    # Add every question as a node
    for index in questions_df.index:
        graph.add_node(index)

    # Add edges using similarity results
    for _, pair in similarity_df.iterrows():

        index_1 = int(pair["index_1"])
        index_2 = int(pair["index_2"])

        score = float(pair["similarity_score"])

        # Validate indexes
        if index_1 not in questions_df.index:
            continue

        if index_2 not in questions_df.index:
            continue

        graph.add_edge(
            index_1,
            index_2,
            weight=score
        )

    clusters = []

    cluster_id = 1

    # Find connected components
    for component in nx.connected_components(graph):

        if len(component) < 2:
            continue

        for index in sorted(component):

            question = questions_df.loc[index]

            clusters.append({

                "cluster_id": cluster_id,

                "question": question["text"],

                "part": question["part"],

                "module": question["module"],

                "year": question["year"],

                "marks": question["marks"],

                "source_file": question["source_file"]

            })

        cluster_id += 1

    cluster_df = pd.DataFrame(clusters)

    if not cluster_df.empty:

        print(
            f"Total repeated-question clusters: "
            f"{cluster_df['cluster_id'].nunique()}"
        )

    return cluster_df
