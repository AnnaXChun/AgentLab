import json
from pathlib import Path

from algorithms import algorithm_a, algorithm_b

config = json.loads(Path("config.json").read_text())
n = config["N"]
a, operations_a = algorithm_a(n)
b, operations_b = algorithm_b(n)
assert a == b
Path("metrics.json").write_text(
    json.dumps(
        {
            "N": n,
            "result_a": a,
            "result_b": b,
            "operations_a": operations_a,
            "operations_b": operations_b,
        },
        sort_keys=True,
    )
)
