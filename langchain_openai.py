"""Lightweight shim providing `OpenAIEmbeddings` and `ChatOpenAI`.

This file makes the repo independent of a particular `langchain` version by
exposing the small subset of APIs used in this project:
- `OpenAIEmbeddings(api_key, base_url, model)` with methods `embed_documents` and `embed_query`.
- `ChatOpenAI(model_name, api_key, base_url)` with method `invoke(messages)` returning an
  object that has a `content` attribute (string) — matching the project's expectations.

It uses the official `openai` package under the hood. Ensure `openai` is in
your `requirements.txt` / environment.
"""
from __future__ import annotations
from types import SimpleNamespace
from typing import List, Optional, Any

import openai


class OpenAIEmbeddings:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        if api_key:
            openai.api_key = api_key
        if base_url:
            # openai Python lib uses api_base for custom endpoints
            openai.api_base = base_url
        self.model = model or "text-embedding-3-small"
        self._client = openai

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents and return list of embeddings."""
        if not texts:
            return []
        resp = self._client.Embedding.create(model=self.model, input=texts)
        return [d["embedding"] for d in resp["data"]]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string and return its embedding vector."""
        resp = self._client.Embedding.create(model=self.model, input=text)
        return resp["data"][0]["embedding"]


class ChatOpenAI:
    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None):
        if api_key:
            openai.api_key = api_key
        if base_url:
            openai.api_base = base_url
        self.model_name = model_name or "gpt-4o-mini"
        self._client = openai

    def invoke(self, messages: List[dict]) -> Any:
        """Call the chat completions endpoint and return an object with `.content`.

        `messages` should be a list of `{"role": "system|user|assistant", "content": "..."}`
        dictionaries (this matches the project's usage).
        """
        resp = self._client.ChatCompletion.create(model=self.model_name, messages=messages)
        # Try to be permissive about the response shape
        try:
            content = resp["choices"][0]["message"]["content"]
        except Exception:
            # Fallback: stringify top-level text if shape differs
            content = str(resp)

        return SimpleNamespace(content=content)
