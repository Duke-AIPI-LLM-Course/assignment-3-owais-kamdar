import numpy as np
import matplotlib.pyplot as plt
from search import retrieve_top_chunks

def plot_similarity_histogram(query, top_k=7):
    """
    Plots a histogram of cosine similarity scores for retrieved chunks.
    """
    retrieved_chunks = retrieve_top_chunks(query, top_k=top_k)

    if not retrieved_chunks:
        print("No relevant chunks retrieved.")
        return

    similarity_scores = [chunk[3] for chunk in retrieved_chunks]  # similarity scores

    # plot histogram
    plt.figure(figsize=(9, 5))
    plt.hist(similarity_scores, bins=10, alpha=0.75, color='red', edgecolor="black")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xlabel("Cosine Similarity Score")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Retrieved Chunk Similarities for Query: {query}")
    plt.show()

