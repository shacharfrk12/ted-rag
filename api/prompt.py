import json
from create_index import (
    get_vector_index,
    PINECONE_INDEX,
    LLMOD_API_KEY,
    LLMOD_URL,
    EMBEDDING_MODEL,
    LLM_MODEL,
)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from query_index import full_query_pipeline

# ⚠️ Globals MAY be reused, but don’t rely on it
_index = None
_embeddings = None
_chat = None


def _init_clients():
    global _index, _embeddings, _chat

    if _index is None:
        _index = get_vector_index(PINECONE_INDEX)

    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            api_key=LLMOD_API_KEY,
            base_url=LLMOD_URL,
            model=EMBEDDING_MODEL,
        )

    if _chat is None:
        _chat = ChatOpenAI(
            model_name=LLM_MODEL,
            api_key=LLMOD_API_KEY,
            base_url=LLMOD_URL,
        )

    return _index, _embeddings, _chat


def handler(request):
    try:
        if request.method != "POST":
            return {
                "statusCode": 405,
                "body": json.dumps({"error": "POST only"}),
            }

        body = json.loads(request.body or "{}")
        question = body.get("question")

        if not question:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'question'"}),
            }

        index, embeddings, chat = _init_clients()

        result = full_query_pipeline(
            question,
            index,
            embeddings,
            "system_prompt.txt",
            chat,
        )

        return {
            "statusCode": 200,
            "body": json.dumps(result),
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
        }