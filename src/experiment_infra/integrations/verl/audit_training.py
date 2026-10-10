"""GPU result acceptance: compare actual tensor values, not checkpoint metadata."""

import hashlib
import json
from pathlib import Path


def tensor_inventory(directory):
    from safetensors import safe_open

    paths = sorted(Path(directory).glob("*.safetensors"))
    if not paths:
        raise ValueError(f"No safetensors weights in {directory}")
    result = {}
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if name in result:
                    raise ValueError("Duplicate tensor key")
                result[name] = path
    return result


def parameter_hash(directory):
    import torch
    from safetensors import safe_open

    digest = hashlib.sha256()
    for name, path in sorted(tensor_inventory(directory).items()):
        with safe_open(path, framework="pt", device="cpu") as f:
            tensor = f.get_tensor(name).contiguous()
            digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def audit_training(directory, initial_model):
    import torch
    from safetensors import safe_open

    from experiment_infra.integrations.verl.summary import summarize

    root = Path(directory)
    checkpoints = sorted(
        (root / "checkpoints").glob("global_step_*"), key=lambda p: int(p.name.rsplit("_", 1)[1])
    )
    if not checkpoints or int(checkpoints[-1].name.rsplit("_", 1)[1]) < 2:
        raise ValueError("At least two completed optimization steps/checkpoints required")
    final = checkpoints[-1] / "actor" / "huggingface"
    before, after = tensor_inventory(initial_model), tensor_inventory(final)
    if set(before) != set(after):
        raise ValueError(
            "Initial/final tensor inventories differ; cannot certify parameter changes"
        )
    changed = []
    for name in sorted(before):
        with (
            safe_open(before[name], framework="pt", device="cpu") as a,
            safe_open(after[name], framework="pt", device="cpu") as b,
        ):
            x, y = a.get_tensor(name), b.get_tensor(name)
            if x.shape != y.shape or not torch.isfinite(y).all():
                raise ValueError("Invalid final parameters")
            if not torch.equal(x.to(y.dtype), y):
                changed.append(name)
    initial = json.loads((root / "initial_policy.json").read_text())
    initial_hash, final_hash = parameter_hash(initial_model), parameter_hash(final)
    if initial["parameter_sha256"] != initial_hash:
        raise ValueError("Initial checkpoint was modified during training")
    if not changed or initial_hash == final_hash:
        raise ValueError("No verified policy parameter update")
    stats = summarize(root)
    train = [s for s in stats if s["split"] == "train"]
    val = sorted([s for s in stats if s["split"] == "test"], key=lambda s: s["training_step"])
    if not train or len(val) < 2 or val[-1]["training_step"] <= val[0]["training_step"]:
        raise ValueError("Missing online training or before/after evaluation")
    report = {
        "verified": True,
        "initial_checkpoint_hash": initial_hash,
        "final_checkpoint_hash": final_hash,
        "parameters_changed": len(changed),
        "training_steps": int(checkpoints[-1].name.rsplit("_", 1)[1]),
        "before_success_rate": val[0]["num_success"] / val[0]["num_rollouts"],
        "after_success_rate": val[-1]["num_success"] / val[-1]["num_rollouts"],
        "number_of_rollouts": sum(s["num_rollouts"] for s in train),
        "number_of_valid_rollouts": sum(s["valid_policy_traces"] for s in train),
        "training_reward_curve": [
            {"step": s["training_step"], "reward": s["avg_reward"]} for s in train
        ],
    }
    (root / "policy_update_evidence.json").write_text(json.dumps(report, indent=2))
    return report
