#evaltest.py

from eval import plot_results, evaluate_rag

if __name__ == "__main__":
    # test queries
    queries = [
        "What are the symptoms of depression?",
        "How is depression treated?",
        "What are common risk factors for depression?"
    ]
    
    results = [evaluate_rag(q) for q in queries]

    # print thorough responses
    for res in results:
        print("\n" + "=" * 100)
        print(f"**Query:** {res['Query']}")
        print("**Baseline Response:**\n" + "-" * 50)
        print(res["Baseline Response"])
        print("**RAG Response:**\n" + "-" * 50)
        print(res["RAG Response"])
        print("**Evaluation Scores:**")
        print(f"  - Jaccard Similarity: Baseline = {res['Baseline Jaccard']:.3f}, RAG = {res['RAG Jaccard']:.3f}")
        print(f"  - Response Length: Baseline = {res['Baseline Length']} chars, RAG = {res['RAG Length']} chars")
        print(f"  - LLM GPT-4o Scores (1-10):")
        print(f"    - Relevance:  Baseline = {res['Baseline Relevance']:.1f}, RAG = {res['RAG Relevance']:.1f}")
        print(f"    - Completeness:  Baseline = {res['Baseline Completeness']:.1f}, RAG = {res['RAG Completeness']:.1f}")
        print(f"    - Conciseness:  Baseline = {res['Baseline Conciseness']:.1f}, RAG = {res['RAG Conciseness']:.1f}")
        print(f"    - **Overall GPT-4o Score:** Baseline = {res['Baseline Score']:.1f}, RAG = {res['RAG Score']:.1f}")

    # Generate plots
    plot_results(results)
