"""Run pinned SAB Docker harness functions without altering native scoring.

Separate process: upstream imports and build state stay out of the batch runtime.
No keys in arguments/images; visualization judge tasks need an audited evaluator setup.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch


def protect_native_inputs(create):
    """Keep official scorer logic intact while mounting its inputs read-only."""

    def guarded(collection, *args, **kwargs):
        volumes = kwargs.get("volumes", {})
        kwargs["volumes"] = {
            source: {**options, "mode": "ro"}
            if options.get("bind")
            in {
                "/testbed/benchmark",
                "/testbed/pred_programs",
                "/testbed/compute_scores.py",
                "/testbed/gpt4_visual_judge.py",
            }
            else dict(options)
            for source, options in volumes.items()
        }
        return create(collection, *args, **kwargs)

    return guarded


def main():
    # Avoid our sibling evaluation.py shadowing upstream evaluation/ namespace.
    sys.path = [p for p in sys.path if Path(p).resolve() != Path(__file__).parent]
    import docker
    from docker.models.containers import ContainerCollection
    from evaluation.harness.docker_build import build_base_images
    from evaluation.harness.run_evaluation import run_instances
    from evaluation.harness.test_spec import make_test_spec

    task_file, data, predictions, run_id = sys.argv[1:]
    task = json.loads(Path(task_file).read_text())
    client = docker.from_env()
    try:
        build_base_images(client, [task], data, predictions, False)
        with patch.object(
            ContainerCollection, "create", protect_native_inputs(ContainerCollection.create)
        ):
            run_instances([task], data, predictions, "instance", False, True, 1, run_id, 1800)
        spec = make_test_spec(task, data, predictions)
        image = client.images.get(spec.instance_image_key)
        Path("evaluation_environment.json").write_text(
            json.dumps(
                {
                    "image_id": image.id,
                    "architecture": image.attrs.get("Architecture"),
                    "base_image_id": client.images.get(spec.base_image_key).id,
                }
            )
        )
    finally:
        # Official run_instance normally cleans up; cover timeout/error leftovers too.
        for container in client.containers.list(
            all=True, filters={"name": f"sab.eval.{task['instance_id']}.{run_id}"}
        ):
            container.remove(force=True)
        client.close()


if __name__ == "__main__":
    main()
