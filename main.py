
from extractor import extract_data, clean_and_organize_text, clean_toc
from processing import chunk_text
from embeddings import generate_embeddings
from search import retrieve_top_chunks
from visualize import plot_similarity_histogram
from storing import store_embeddings_in_db, connect_db
from search import query_llm, retrieve_top_chunks

# set pdf path
pdf_path = "depression.pdf"

def test_extraction():
    """test text extraction from pdf."""
    metadata, raw_toc, raw_text = extract_data(pdf_path)
    
    # clean extracted content
    cleaned_toc = clean_toc(raw_toc)
    cleaned_text = clean_and_organize_text(raw_text)

    # print results
    print("\nmetadata:", metadata)
    print("\ncleaned toc:", cleaned_toc)  # preview first 5
    print("\nfirst 500 characters of cleaned text:\n", cleaned_text[:500])

    return cleaned_toc, cleaned_text


def test_chunking(cleaned_text, cleaned_toc):
    """test text chunking after cleaning."""
    chunks = chunk_text(cleaned_text, cleaned_toc, max_chunk_size=800, overlap=100)

    # print preview of results
    print(f"total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:5]):  # first 5 chunks
        print(f"\nchunk {i + 1} ({len(chunk['content'])} chars) - heading: {chunk['heading']}")
        print(f"chunk id: {chunk['chunk_id']}")
        print(f"{'-' * 50}\n{chunk['content']}\n{'-' * 50}")

    return chunks

def test_embeddings(chunks):
    """test embedding generation for text chunks."""
    chunks_with_embeddings = generate_embeddings(chunks)

    # preview results
    if chunks_with_embeddings:
        print(f"generated embeddings for {len(chunks_with_embeddings)} chunks")
        for i, chunk in enumerate(chunks_with_embeddings[:3]):  # first 3 chunks
            print(f"\nchunk {i + 1} - heading: {chunk['heading']}")
            print(f"chunk id: {chunk['chunk_id']}")
            print(f"content ({len(chunk['content'])} characters): {chunk['content'][:100]}...")
            print(f"embedding (first 5 values): {chunk['embedding'][:5]}...")
    else:
        print("no embeddings generated.")

    return chunks_with_embeddings

# test db connection
def test_db_connection():
    """test Supabase database connection."""
    conn = connect_db()
    if conn:
        print("Database connection successful.")
        conn.close()
    else:
        print("Database connection failed.")


def test_store_embeddings(chunks_with_embeddings):
    """test storing embeddings into Supabase."""
    try:
        store_embeddings_in_db(chunks_with_embeddings)
        print("Embeddings successfully stored in the database.")
    except Exception as e:
        print(f"Error storing embeddings: {e}")

# test retrieval of chunks
def test_retrieval(query):
    """test retrieval of top relevant chunks based on query."""
    top_chunks = retrieve_top_chunks(query)

    if top_chunks:
        print("top retrieved chunks:")
        for chunk in top_chunks:
            print(f"chunk id: {chunk[0]} - {chunk[1]}")
            print(f"similarity score: {chunk[3]:.4f}")
            print(f"content preview: {chunk[2][:200]}...\n{'-' * 50}")
    else:
        print("no relevant chunks found.")

# test query
def test_llm_query(query):
    baseline_response = query_llm(query)  # No RAG
    top_chunks = retrieve_top_chunks(query)
    rag_response = query_llm(query, retrieved_chunks=top_chunks) # with RAG
    print("Baseline Response (No RAG):")
    print(baseline_response)

    print("RAG-Enhanced Response:")
    print(rag_response)


def test_visualization(query):
    """test similarity score visualization."""
    plot_similarity_histogram(query, top_k=7)


if __name__ == "__main__":
    # run tests
    cleaned_toc, cleaned_text = test_extraction()
    chunks = test_chunking(cleaned_text, cleaned_toc)
    chunks_with_embeddings = test_embeddings(chunks)
    test_db_connection()

    test_store_embeddings(chunks_with_embeddings)

    # test query
    query = "What are the symptoms of depression?"

    test_retrieval(query)
    test_llm_query(query)
    test_visualization(query)
