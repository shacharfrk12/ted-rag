"""GET /api/stats endpoint handler.

Returns current RAG hyperparameters (chunk_size, overlap_ratio, top_k).
"""

from http.server import BaseHTTPRequestHandler
import json
import sys
from pathlib import Path
from Constants import CHUNK_SIZE, OVERLAP_RATIO, TOP_K



class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        response = {
            "chunk_size": CHUNK_SIZE,
            "overlap_ratio": OVERLAP_RATIO,
            "top_k": TOP_K
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())