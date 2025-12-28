from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from create_index import get_vector_index, PINECONE_INDEX, LLMOD_API_KEY, LLMOD_URL, EMBEDDING_MODEL, LLM_MODEL
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from query_index import prepare_context_for_qa, run_query_in_model, format_rag_response, TOP_K, MAX_CHUNKS_PER_TALK

app = FastAPI(title="RAG Project API")


class PromptRequest(BaseModel):
    question: str


# Lazy-initialized clients (kept global to reuse between requests)
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
    """Accepts JSON {"question": "..."} and returns RAG-formatted JSON."""
    try:
        index, embeddings_model, chat_model = _init_clients()

        # Retrieve context and matching chunks
        context_str, retrieved_chunks = prepare_context_for_qa(index, embeddings_model, req.question, TOP_K, MAX_CHUNKS_PER_TALK)

        # Run chat model
        response_text, system_prompt = run_query_in_model(context_str, "system_prompt.text", req.question, chat_model)

        # Format into the required output schema
        output = format_rag_response(response_text, retrieved_chunks, system_prompt, req.question)

        return output

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
