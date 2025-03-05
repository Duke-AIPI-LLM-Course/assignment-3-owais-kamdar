#eval.py

import re
import numpy as np
import matplotlib.pyplot as plt
from search import retrieve_top_chunks, query_llm

def jaccard_similarity(text1, text2):
    """
    Computes Jaccard similarity between two texts (word-level comparison).
    """
    words1, words2 = set(text1.lower().split()), set(text2.lower().split())
    return len(words1 & words2) / len(words1 | words2) if words1 | words2 else 0

def extract_scores(text):
    """
    Numerical scores for Baseline and RAG:
    - Relevance
    - Completeness
    - Conciseness
    - Calculates the overall score by averaging the three metrics
    """
    relevance_baseline = re.search(r"Baseline Relevance:\s*([\d.]+)", text)
    completeness_baseline = re.search(r"Baseline Completeness:\s*([\d.]+)", text)
    conciseness_baseline = re.search(r"Baseline Conciseness:\s*([\d.]+)", text)

    relevance_rag = re.search(r"RAG Relevance:\s*([\d.]+)", text)
    completeness_rag = re.search(r"RAG Completeness:\s*([\d.]+)", text)
    conciseness_rag = re.search(r"RAG Conciseness:\s*([\d.]+)", text)

    baseline_relevance = float(relevance_baseline.group(1)) if relevance_baseline else None
    baseline_completeness = float(completeness_baseline.group(1)) if completeness_baseline else None
    baseline_conciseness = float(conciseness_baseline.group(1)) if conciseness_baseline else None

    rag_relevance = float(relevance_rag.group(1)) if relevance_rag else None
    rag_completeness = float(completeness_rag.group(1)) if completeness_rag else None
    rag_conciseness = float(conciseness_rag.group(1)) if conciseness_rag else None

    # average scores
    mean_baseline = (
        (baseline_relevance + baseline_completeness + baseline_conciseness) / 3
    )

    mean_rag = (
        (rag_relevance + rag_completeness + rag_conciseness) / 3
    )

    return (
        mean_baseline, mean_rag, 
        baseline_relevance, baseline_completeness, baseline_conciseness,
        rag_relevance, rag_completeness, rag_conciseness
    )

def evaluate_response_quality(baseline, rag):
    """
    Uses GPT-4o as a judge to rate the responses on:
    - Relevance
    - Completeness
    - Conciseness
    """
    prompt = f"""You are an expert evaluator. Rate the two responses based on:

    - **Relevance**: Does the response correctly address the question? (1-10)
    - **Completeness**: Does it provide enough useful details? (1-10)
    - **Conciseness**: Is it direct and to the point without unnecessary filler? (1-10)

    **Baseline Response:** 
    {baseline}

    **RAG Response:**
    {rag}

    Provide scores in this format:

    Baseline Relevance: A 
    Baseline Completeness: B  
    Baseline Conciseness: C  
    RAG Relevance: D
    RAG Completeness: E  
    RAG Conciseness: F  
    """
    
    eval_response = query_llm(prompt)

    # extract scores
    return extract_scores(eval_response)

def evaluate_rag(query, top_k=7):
    """
    Evaluates the effectiveness of RAG vs. non-RAG responses.
    """
    # retrieve relevant chunks
    retrieved_chunks = retrieve_top_chunks(query, top_k=top_k)
    retrieved_text = "\n\n".join([f"{chunk[1]}: {chunk[2]}" for chunk in retrieved_chunks]) if retrieved_chunks else ""
    
    # LLM responses
    baseline_response = query_llm(query)
    rag_response = query_llm(query, retrieved_chunks=retrieved_chunks)

    # Jaccard similarity
    jaccard_baseline = jaccard_similarity(retrieved_text, baseline_response)
    jaccard_rag = jaccard_similarity(retrieved_text, rag_response)

    # response length
    baseline_length = len(baseline_response.split())
    rag_length = len(rag_response.split())


    # evaluate response using llm
    (
        mean_baseline, mean_rag, 
        baseline_relevance, baseline_completeness, baseline_conciseness,
        rag_relevance, rag_completeness, rag_conciseness
    ) = evaluate_response_quality(baseline_response, rag_response)

    return {
        "Query": query,
        "Baseline Response": baseline_response,
        "RAG Response": rag_response,
        "Baseline Length": baseline_length,
        "RAG Length": rag_length,
        "Baseline Jaccard": jaccard_baseline,
        "RAG Jaccard": jaccard_rag,
        "Baseline Relevance": baseline_relevance,
        "Baseline Completeness": baseline_completeness,
        "Baseline Conciseness": baseline_conciseness,
        "RAG Relevance": rag_relevance,
        "RAG Completeness": rag_completeness,
        "RAG Conciseness": rag_conciseness,
        "Baseline Score": mean_baseline,
        "RAG Score": mean_rag
    }


def plot_results(results):
    """
    Generates comparison graph for Baseline vs. RAG,
    displaying individual scores and overlaying the average score.
    """
    # missing scores
    valid_results = [res for res in results if res["Baseline Score"] is not None and res["RAG Score"] is not None]

    for res in valid_results:
        query = res["Query"]

        # get scores
        baseline_relevance, rag_relevance = res["Baseline Relevance"], res["RAG Relevance"]
        baseline_completeness, rag_completeness = res["Baseline Completeness"], res["RAG Completeness"]
        baseline_conciseness, rag_conciseness = res["Baseline Conciseness"], res["RAG Conciseness"]
        baseline_avg, rag_avg = res["Baseline Score"], res["RAG Score"]

        # locations for bars
        x_baseline = np.array([0, 1, 2])  # Baseline metrics
        x_rag = np.array([4, 5, 6])  # RAG metrics

        metrics = ["Relevance", "Completeness", "Conciseness"]
        baseline_scores = [baseline_relevance, baseline_completeness, baseline_conciseness]
        rag_scores = [rag_relevance, rag_completeness, rag_conciseness]

        colors = ["skyblue", "blue", "darkblue"]  # baseline colors
        rag_colors = ["lightgreen", "green", "darkgreen"]  # rag colors

        fig, ax = plt.subplots(figsize=(12, 6))

        # baseline bars
        for i in range(3):
            ax.bar(x_baseline[i], baseline_scores[i], color=colors[i], width=0.6, label=f"Baseline {metrics[i]}")
            ax.text(x_baseline[i], baseline_scores[i] + 0.2, f"{baseline_scores[i]:.1f}", ha='center', fontsize=10, fontweight="bold")

        # RAG bars
        for i in range(3):
            ax.bar(x_rag[i], rag_scores[i], color=rag_colors[i], width=0.6, label=f"RAG {metrics[i]}")
            ax.text(x_rag[i], rag_scores[i] + 0.2, f"{rag_scores[i]:.1f}", ha='center', fontsize=10, fontweight="bold")

        # avg scores overlay
        ax.axhline(y=baseline_avg, color="blue", linestyle="dashed", linewidth=1.5, label=f"Baseline Avg: {baseline_avg:.1f}")
        ax.axhline(y=rag_avg, color="green", linestyle="dashed", linewidth=1.5, label=f"RAG Avg: {rag_avg:.1f}")

        # formatting
        ax.set_xticks([1, 5])
        ax.set_xticklabels(["No RAG", "RAG"], fontsize=12, fontweight="bold")

        ax.set_ylabel("Score (1-10)")
        ax.set_title(f"Comparison of Baseline vs. RAG\nQuery: \"{query}\"", fontsize=14)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.show()

