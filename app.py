import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from search import retrieve_top_chunks, query_llm
from eval import evaluate_rag

# ------------------ Streamlit Page Configuration ------------------
st.set_page_config(page_title="RAG Depression Guide", layout="wide")

st.title("RAG System for Depression Guide")
st.write("Enter a query, and the AI will generate answers using both **retrieved document context (RAG)** and **baseline LLM only**.")

# ------------------ User Query Input ------------------
st.subheader("Step 1: Enter Your Question")
query = st.text_input("Enter your question:", placeholder="e.g., What are the symptoms of depression?")


if st.button("Search"):
    if query.strip():
        with st.spinner("🔍 Searching for relevant information..."):

            # ------------------ Retrieve Relevant Chunks ------------------
            st.subheader("Step 2: Retrieving Relevant Document Context")
            top_chunks = retrieve_top_chunks(query, top_k=10)

            if top_chunks:
                for chunk_id, heading, content, similarity in top_chunks:
                    with st.expander(f"{heading} (Similarity: {similarity:.4f})"):
                        st.write(f"**Chunk ID:** {chunk_id}")
                        st.write(f"**Content:** {content}")

                # ------------------ Visualizing Similarity Scores ------------------
                st.subheader("Step 3: Similarity Score Histogram")

                # create a similarity histogram in Streamlit
                similarity_scores = [chunk[3] for chunk in top_chunks]
                fig, ax = plt.subplots(figsize=(9, 5))
                ax.hist(similarity_scores, bins=10, alpha=0.75, color='red', edgecolor='black')
                ax.grid(axis="y", linestyle="--", alpha=0.6)
                ax.set_xlabel("Cosine Similarity Score")
                ax.set_ylabel("Count")
                ax.set_title(f"Histogram of Retrieved Chunk Similarities for Query: {query}")
                st.pyplot(fig)

                # ------------------ Generate Responses ------------------
                st.subheader("Step 4: AI-Generated Answers")

                baseline_response = query_llm(query)  # No RAG
                rag_response = query_llm(query, retrieved_chunks=top_chunks)  # With RAG

                st.write("#### **Baseline (LLM Only)**")
                st.info(baseline_response)

                st.write("#### **RAG-Enhanced Response (LLM + Retrieved Context)**")
                st.success(rag_response)

                # ------------------ Evaluate Responses ------------------
                st.subheader("Step 5: Evaluation of Responses")

                eval_results = evaluate_rag(query)

                # display evaluation metrics in a table
                eval_df = pd.DataFrame({
                    "Metric": ["Response Length (characters)", "Jaccard Similarity", "Average GPT-4o Score"],
                    "Baseline": [eval_results["Baseline Length"], eval_results["Baseline Jaccard"], eval_results["Baseline Score"]],
                    "RAG": [eval_results["RAG Length"], eval_results["RAG Jaccard"], eval_results["RAG Score"]]
                })

                st.table(eval_df)

                # ------------------ Plotting Evaluation Metrics ------------------
                st.subheader("Step 6: Comparative Visualizations")

                # extract individual scores
                baseline_relevance = eval_results["Baseline Relevance"]
                rag_relevance = eval_results["RAG Relevance"]

                baseline_completeness = eval_results["Baseline Completeness"]
                rag_completeness = eval_results["RAG Completeness"]

                baseline_conciseness = eval_results["Baseline Conciseness"]
                rag_conciseness = eval_results["RAG Conciseness"]

                baseline_avg = eval_results["Baseline Score"]
                rag_avg = eval_results["RAG Score"]


                # X locations for bars
                x_baseline = np.array([0, 1, 2])  # baseline Relevance, Completeness, Conciseness
                x_rag = np.array([4, 5, 6])  # RAG Relevance, Completeness, Conciseness

                # categories and their positions
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

                # average score overlay
                ax.axhline(y=baseline_avg, color="blue", linestyle="dashed", linewidth=1.5, label=f"Baseline Avg: {baseline_avg:.1f}")
                ax.axhline(y=rag_avg, color="green", linestyle="dashed", linewidth=1.5, label=f"RAG Avg: {rag_avg:.1f}")

                # format the graph
                ax.set_xticks([1, 5])
                ax.set_xticklabels(["No RAG", "RAG"], fontsize=12, fontweight="bold")

                ax.set_ylabel("Score (1-10)")
                ax.set_title(f"Comparison of Baseline vs. RAG\nQuery: \"{query}\"", fontsize=14)
                ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

                plt.tight_layout()
                st.pyplot(fig)

            else:
                st.warning("No relevant chunks found.")

    else:
        st.error("Please enter a query before searching.")

