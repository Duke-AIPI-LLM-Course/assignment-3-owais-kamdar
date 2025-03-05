# storing.py

import psycopg2
import json
import numpy as np
from dotenv import load_dotenv
import os


# load env
load_dotenv()



def connect_db():
    """
    Connect to Supabase PostgreSQL using session pooler.
    """
    try:
        print("connecting to supabase..")
        conn = psycopg2.connect(os.getenv("postgres"))
        print("connection successful")
        return conn
    except Exception as e:
        print(f"Error connecting to Supabase: {e}")
        return None

def store_embeddings_in_db(chunks_with_embeddings):
    """
    Stores chunk embeddings into a PostgreSQL database.
    """
    conn = connect_db()
    if not conn:
        return

    try:
        cur = conn.cursor()

        for chunk in chunks_with_embeddings:
            chunk_id = chunk["chunk_id"]
            heading = chunk["heading"]
            content = chunk["content"]
            vector = np.array(chunk["embedding"], dtype=np.float32).tolist()  # vector format



            cur.execute(
                """
                INSERT INTO embeddings (chunk_id, heading, content, vector)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (chunk_id)
                DO UPDATE 
                SET heading = EXCLUDED.heading, content = EXCLUDED.content, vector = EXCLUDED.vector
                """,
                (chunk_id, heading, content, vector)
            )

        conn.commit()
        print("Embeddings stored successfully in PostgreSQL")

    except Exception as e:
        print(f"Error storing embeddings: {e}")

    finally:
        cur.close()
        conn.close()

