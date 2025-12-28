"""GET /api/stats endpoint handler.

Returns current RAG hyperparameters (chunk_size, overlap_ratio, top_k).
"""

from http.server import BaseHTTPRequestHandler
import json
import sys
from pathlib import Path


class handler(BaseHTTPRequestHandler):
    """HTTP handler for reporting current RAG hyperparameters."""
    def do_GET(self):
        """Return chunking and retrieval settings as JSON."""
        response = {
            "chunk_size": 0,
            "overlap_ratio": 0,
            "top_k": 0,
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())