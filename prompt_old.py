from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from create_index import get_vector_index, PINECONE_INDEX, LLMOD_API_KEY, LLMOD_URL, EMBEDDING_MODEL, LLM_MODEL
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from query_index import full_query_pipeline, TOP_K, MAX_CHUNKS_PER_TALK
import os
from Constants import CHUNK_SIZE, OVERLAP_RATIO, TOP_K


app = FastAPI()

class PromptRequest(BaseModel):
    question: str

_index = None
_embeddings = None
_chat = None

def _init_clients():
    global _index, _embeddings, _chat
    if _index is None:
        _index = get_vector_index(PINECONE_INDEX)
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(api_key=LLMOD_API_KEY, base_url=LLMOD_URL, model=EMBEDDING_MODEL)
    if _chat is None:
        _chat = ChatOpenAI(model_name=LLM_MODEL, api_key=LLMOD_API_KEY, base_url=LLMOD_URL)
    return _index, _embeddings, _chat

@app.post("/api/prompt")
async def api_prompt(req: PromptRequest):
    try:
        index, embeddings_model, chat_model = _init_clients()
        return full_query_pipeline(req.question, index, embeddings_model, "system_prompt.txt", chat_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

    
@app.get("/api/stats")
async def api_stats():
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    }
