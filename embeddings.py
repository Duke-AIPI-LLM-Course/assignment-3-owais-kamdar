# embeddings.py

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("openai_key"))

def generate_embeddings(chunks, model="text-embedding-ada-002"):
    """
    Generates embeddings for each chunk using OpenAI's embedding API.
    """

    texts = [chunk["content"] for chunk in chunks]

    # create embeddings
    try:
        response = client.embeddings.create(input=texts, model=model)
        embedded_chunks = [
            {
                "chunk_id": chunk["chunk_id"],
                "heading": chunk["heading"],
                "content": chunk["content"],
                "embedding": response.data[i].embedding
            }
            for i, chunk in enumerate(chunks)
        ]
        return embedded_chunks

    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []  # return empty list if error

