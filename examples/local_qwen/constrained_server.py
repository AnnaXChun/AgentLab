"""Loopback-only, single-request MLX server with Outlines JSON Schema decoding."""

import argparse
import json
import time
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, HTTPServer
from uuid import uuid4

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from outlines.backends.outlines_core import OutlinesCoreBackend
from outlines.models.mlxlm import MLXLM


class LocalModel:
    def __init__(self, path, thinking=False):
        self.path = path
        self.thinking = thinking
        self.model, self.tokenizer = load(path)
        self.backend = OutlinesCoreBackend(MLXLM(self.model, self.tokenizer))

    @lru_cache(maxsize=4)
    def processor(self, schema):
        return self.backend.get_json_schema_logits_processor(schema)

    def complete(self, request):
        if request.get("stream"):
            raise ValueError("Streaming is not supported by this local runner server")
        if request.get("model") not in (self.path, "default_model"):
            raise ValueError("Only the pinned local model is available")
        response_format = request.get("response_format", {})
        processors = None
        if response_format.get("type") == "json_schema":
            schema = response_format["json_schema"]["schema"]
            processor = self.processor(json.dumps(schema))
            processor.reset()
            processors = [processor]
        elif response_format.get("type") not in (None, "json_object"):
            raise ValueError("Unsupported response format")
        prompt = self.tokenizer.apply_chat_template(
            request["messages"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )
        limit = request.get("max_tokens", 2048)
        if self.thinking and processors:
            processors = [
                AfterThinking(
                    processors[0],
                    self.tokenizer.convert_tokens_to_ids("</think>"),
                    min(1024, limit // 2),
                )
            ]
        mx.random.seed(request.get("seed", 0))
        sampler = make_sampler(temp=request.get("temperature", 0.7), top_p=0.8, top_k=20)
        chunks = []
        for result in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            sampler=sampler,
            logits_processors=processors,
            max_tokens=limit,
        ):
            chunks.append(result.text)
        content = "".join(chunks)
        if self.thinking:
            # Private reasoning is ephemeral: never return or log it.
            content = content.split("</think>", 1)[1].strip() if "</think>" in content else ""
        return {
            "id": "chatcmpl-" + str(uuid4()),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.path,
            "system_fingerprint": "mlx-0.32.3-outlines-1.3.3-bf16-"
            + ("thinking1024" if self.thinking else "nonthinking"),
            "choices": [
                {
                    "index": 0,
                    "finish_reason": result.finish_reason,
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.generation_tokens,
                "total_tokens": result.prompt_tokens + result.generation_tokens,
            },
        }


class AfterThinking:
    """Allow bounded model-native reasoning, then constrain only the final JSON."""

    def __init__(self, processor, end_token, budget):
        self.processor, self.end_token = processor, end_token
        self.budget, self.final = budget, False
        self.initial_length = None

    def __call__(self, tokens, logits):
        # MLX prefills most of the prompt before invoking logits processors.
        # Count generation relative to the first callback, not the full prompt.
        if self.initial_length is None:
            self.initial_length = len(tokens)
        if self.final or int(tokens[-1]) == self.end_token:
            self.final = True
            return self.processor(tokens, logits)
        if len(tokens) - self.initial_length >= self.budget:
            forced = mx.full_like(logits, -float("inf"))
            forced[..., self.end_token] = 0
            return forced
        return logits


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # Never log prompts, completions or credentials.

    def send_json(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            self.send_json(
                200, {"object": "list", "data": [{"id": self.server.model.path, "object": "model"}]}
            )
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 2_000_000:
                raise ValueError("Invalid request size")
            request = json.loads(self.rfile.read(length))
            response = self.server.model.complete(request)
        except Exception as exc:
            self.send_json(400, {"error": {"type": type(exc).__name__, "message": str(exc)[:300]}})
            return
        self.send_json(200, response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()
    server = HTTPServer(("127.0.0.1", 8081), Handler)
    server.model = LocalModel(args.model, args.thinking)
    print("Pinned MLX model ready on http://127.0.0.1:8081 with JSON Schema decoding", flush=True)
    server.serve_forever()
