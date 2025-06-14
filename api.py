from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# ---- Setup ----
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index and metadata
index = faiss.read_index("index/faiss_index.index")
with open("index/metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# ---- API Models ----
class Link(BaseModel):
    url: str
    text: str

class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None  # Unused for now

class AnswerResponse(BaseModel):
    answer: str
    links: List[Link]

# ---- Helper Function ----
def search_similar(query: str, top_k: int = 2):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype("float32"), top_k)
    results = []
    for i in indices[0]:
        if i < len(metadata):
            results.append(metadata[i])
    return results

# ---- API Endpoint ----
@app.post("/api/", response_model=AnswerResponse)
def answer_question(payload: QuestionRequest):
    query = payload.question
    similar_links = search_similar(query)

    # Dummy answer for now — feel free to add more logic
    return {
        "answer": f"I found some helpful resources for your question: \"{query}\"",
        "links": [
            {"url": url, "text": f"Related link {i+1}"} for i, url in enumerate(similar_links) if url
        ]
    }

# Root test
@app.get("/")
def root():
    return {"message": "Hello from FastAPI"}
