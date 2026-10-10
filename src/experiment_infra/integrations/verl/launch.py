"""Launch the unmodified official trainer on Linux/NVIDIA. No local fallback optimizer."""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from experiment_infra.integrations.verl import VERL_COMMIT


def preflight():
    if platform.system() != "Linux":
        raise RuntimeError(
            "Real veRL GRPO requires Linux/NVIDIA; use local mock/Docker tests on macOS"
        )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA CUDA device is required")
    dist = importlib.metadata.distribution("verl")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != VERL_COMMIT:
        raise RuntimeError(f"Install the pinned veRL extra at {VERL_COMMIT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/verl/toy_grpo.yaml"))
    parser.add_argument("--model", required=True, help="Local HF checkpoint directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    args, overrides = parser.parse_known_args()
    preflight()
    from omegaconf import OmegaConf

    from experiment_infra.integrations.verl.audit_training import audit_training, parameter_hash
    from experiment_infra.integrations.verl.dataset import prepare

    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    config_path = args.config.resolve()
    config = OmegaConf.load(config_path)
    # Resolve bridge selection only; full config composition remains official Hydra's job.
    config = OmegaConf.merge(OmegaConf.load(config_path.with_name("_grpo_base.yaml")), config)
    config = OmegaConf.merge(config, OmegaConf.from_dotlist([x.lstrip("+") for x in overrides]))
    bridge = config.experiment_bridge
    model = str(Path(args.model).resolve())
    initial_hash = parameter_hash(model)
    (root / "initial_policy.json").write_text(
        json.dumps({"path": model, "parameter_sha256": initial_hash}, indent=2)
    )
    prepare(bridge.benchmark, root / "data", bridge.train_tasks, bridge.val_tasks)
    environment = {
        **os.environ,
        "VERL_SESSION_ID": uuid4().hex,
        "VERL_OUTPUT_DIR": str(root),
        "VERL_MODEL_PATH": model,
        "VERL_INITIAL_CHECKPOINT": "sha256:" + initial_hash,
        "VERL_AGENT_CONFIG": str(config_path.with_name("agent_loops.yaml")),
    }
    command = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        "--config-path",
        str(config_path.parent),
        "--config-name",
        config_path.stem,
        *overrides,
    ]
    (root / "launch.json").write_text(
        json.dumps(
            {
                "command": command,
                "verl_commit": VERL_COMMIT,
                "session_id": environment["VERL_SESSION_ID"],
            },
            indent=2,
        )
    )
    with (root / "trainer.log").open("w") as log:
        subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(json.dumps(audit_training(root, model), indent=2))


if __name__ == "__main__":
    main()
