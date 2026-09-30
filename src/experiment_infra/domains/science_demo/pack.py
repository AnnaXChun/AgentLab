from experiment_infra.core.schema import DomainPack

PACK = DomainPack(
    name="science_demo",
    artifact_types=["source_code", "config", "metric", "log", "checkpoint"],
    verifier=["metric", "test"],
    templates={
        "hypothesis": "Algorithm B uses fewer addition operations than Algorithm A while computing the same sum for N positive integers.",
        "success_criteria": {
            "metric": {"field": "operations_b", "operator": "lt", "baseline_field": "operations_a"},
            "test": {"argv": ["python", "-m", "unittest", "-v", "test_algorithms.py"]},
        },
    },
)
