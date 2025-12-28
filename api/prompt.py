import json
from http.server import BaseHTTPRequestHandler

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


# Globals (persist while server runs)
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


class handler(BaseHTTPRequestHandler):
    """HTTP handler for RAG prompt requests."""

    def do_POST(self):
        """Handle RAG prompt requests."""

        if self.path != "/api/prompt":
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Not found"}).encode())
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body or "{}")

            question = data.get("question")
            if not question:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"error": "Missing 'question'"}).encode()
                )
                return

            index, embeddings, chat = _init_clients()

            result = full_query_pipeline(
                question,
                index,
                embeddings,
                "system_prompt.txt",
                chat,
            )

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())



