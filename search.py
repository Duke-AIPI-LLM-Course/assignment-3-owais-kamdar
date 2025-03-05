#search.py

import numpy as np
from openai import OpenAI
from embeddings import generate_embeddings
from storing import connect_db
from dotenv import load_dotenv
import os
import ast

load_dotenv()

client = OpenAI(api_key=os.getenv("openai_key"))


def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors.
    """
    vec1 = np.array(vec1, dtype=np.float32)
    vec2 = np.array(vec2, dtype=np.float32)

    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))



def query_llm(prompt, retrieved_chunks=None):
    """
    Sends query with context to an LLM (GPT-4o) and returns the response.
    """
    try:
        # rag enabled: inject retrieved document chunks into the prompt
        if retrieved_chunks:
            context = "\n\n".join([f"{chunk[1]}: {chunk[2]}" for chunk in retrieved_chunks])            
            full_prompt = full_prompt = f"Here is relevant document context:\n\n{context}\n\n" \
              f"Use this information to supplement your general knowledge and provide the most complete answer possible. " \
              f"User Query: {prompt}"

        else:
            full_prompt = prompt  # No-RAG case

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an assistant providing information based on retrieved documents."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content

    except Exception as e:
        print(f"Error querying LLM: {e}")
        return f"Error: {e}"  # Return error for debugging


def retrieve_top_chunks(query, top_k=5):
    """
    Retrieves the most relevant chunks based on cosine similarity.
    """
    try:
        # get embedding for the query
        query_embedding = generate_embeddings([{"chunk_id": "query", "heading": "query", "content": query}])
        
        if not query_embedding or "embedding" not in query_embedding[0]:
            print("Failed to generate query embedding.")
            return []

        query_embedding = np.array(query_embedding[0]["embedding"], dtype=np.float32)

        conn = connect_db()
        if not conn:
            print("Database connection failed.")
            return []

        cur = conn.cursor()

        # fetch all stored embeddings
        cur.execute("SELECT chunk_id, heading, content, vector FROM embeddings")
        results = cur.fetchall()

        if not results:
            print("No data found in embeddings table.")
            return []

        similarities = []
        for chunk_id, heading, content, vector in results:
            vector_array = np.array(ast.literal_eval(vector), dtype=np.float32) 
            similarity = cosine_similarity(query_embedding, vector_array)
            similarities.append((chunk_id, heading, content, similarity))

        # sort results by similarity score
        similarities.sort(key=lambda x: x[3], reverse=True)

        return similarities[:top_k]

    except Exception as e:
        print(f"Error retrieving top chunks: {e}")
        return []

    finally:
        if conn:
            conn.close()

