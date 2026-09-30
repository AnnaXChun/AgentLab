"""Prepare or serve pinned Qwen3-1.7B on Apple Silicon without cloud inference."""

import argparse
import json
import os
import platform
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / ".local-model"
MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def prepare():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("This MLX setup requires Apple Silicon macOS")
    python = LOCAL / "venv/bin/python"
    if not python.exists():
        subprocess.run(["uv", "venv", "--python", "3.12", str(LOCAL / "venv")], check=True)
    subprocess.run(
        [
            "uv",
            "pip",
            "sync",
            "--python",
            str(python),
            str(Path(__file__).with_name("requirements.txt")),
        ],
        check=True,
    )
    subprocess.run(
        [
            str(python),
            "-c",
            "from huggingface_hub import snapshot_download; from pathlib import Path; "
            "p=snapshot_download(" + repr(MODEL) + ",revision=" + repr(REVISION) + ","
            "allow_patterns=['*.json','*.safetensors','*.jinja','*.txt','*.model']); "
            "Path(" + repr(str(LOCAL / "model-path.txt")) + ").write_text(p+'\\n')",
        ],
        check=True,
    )
    model_path = (LOCAL / "model-path.txt").read_text().strip()
    (ROOT / ".env.local-qwen").write_text(
        "export MODEL_BASE_URL=http://127.0.0.1:8081/v1\n"
        "export MODEL_API_KEY=''\n"
        "export MODEL_NAME=" + shlex.quote(model_path) + "\n"
    )
    (LOCAL / "model-manifest.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "revision": REVISION,
                "weights": "original BF16, unquantized",
                "engine": "mlx-lm",
                "enable_thinking": False,
                "top_p": 0.8,
                "top_k": 20,
                "path": model_path,
                "structured_output": "prompt plus strict client validation; no server grammar",
            },
            indent=2,
        )
        + "\n"
    )
    print("Prepared. Start with: python examples/local_qwen/manage.py serve")
    print("In another terminal: source .env.local-qwen")


def serve():
    model_path = (LOCAL / "model-path.txt").read_text().strip()
    binary = str(LOCAL / "venv/bin/mlx_lm.server")
    args = [
        binary,
        "--model",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        "8081",
        "--chat-template-args",
        '{"enable_thinking":false}',
        "--temp",
        "0.7",
        "--top-p",
        "0.8",
        "--top-k",
        "20",
        "--prompt-cache-size",
        "1",
        "--prompt-cache-bytes",
        "2GB",
        "--decode-concurrency",
        "1",
        "--prompt-concurrency",
        "1",
        "--log-level",
        "WARNING",
    ]
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}
    os.execve(binary, args, env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "serve"])
    args = parser.parse_args()
    prepare() if args.command == "prepare" else serve()
