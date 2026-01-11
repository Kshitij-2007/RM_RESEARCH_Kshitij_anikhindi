from google import genai
from pydantic import BaseModel
from typing import Literal, List
from pydantic import ValidationError
import numpy as np
import json
import time
import logging
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s")
# adding schema
class Schema(BaseModel):
    topic: str
    answer: List[str]
    validity: Literal["high", "medium", "low"]

client = genai.Client()
# document
documents = [
    "A trainee parachutist must complete five successful static-line jumps to graduate.",
    "The Indian Army Parachute Training School is located in Agra.",
    "Retrieval Augmented Generation reduces hallucinations by grounding answers in retrieved documents.",
    "RAG retrieves relevant information before generating a response."
]
# tokenizer download
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# token making
tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]

w2v_model = Word2Vec(
    sentences=tokenized_docs,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4
)
def embed_document(text):
    tokens = word_tokenize(text.lower())
    vectors = []

    for word in tokens:
        if word in w2v_model.wv:
            vectors.append(w2v_model.wv[word])

    if not vectors:
        return np.zeros(w2v_model.vector_size)

    return np.mean(vectors, axis=0)

# embed all documents
doc_embeddings = np.array([embed_document(doc) for doc in documents])

# semantic search
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
# input
input_user = input("You: ")
logging.info("User input received")
# retrieval
query_embedding = embed_document(input_user)
scores = [cosine_similarity(query_embedding, d) for d in doc_embeddings]
k = 2
top_k_indices = np.argsort(scores)[-k:][::-1]
retrieved_docs = [documents[i] for i in top_k_indices]
context = "\n".join(retrieved_docs)
logging.info("Retrieved %d relevant documents", k)
# prompt structuring
prompt = f"""
You are an assistant.

Use ONLY the information provided in the context below.
If the answer is not present, say you don't know.

Give a step-by-step numbered answer.

Context:
{context}

Question:
{input_user}

Return the response strictly in JSON format.
"""
# generation with retries
for attempt in range(3):
    logging.info("Attempt %d started", attempt + 1)
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": Schema
            }
        )

        if not response.text:
            logging.warning("Empty response from model")
            raise ValueError("Empty response")

        raw_json = json.loads(response.text)
        validated = Schema(**raw_json)

        logging.info("Schema validation successful")
        print("\nValidated Output:")
        print(validated)

        break

    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        logging.error("Attempt %d failed: %s", attempt + 1, e)
        time.sleep(2)