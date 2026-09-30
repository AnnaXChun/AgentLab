# Project workflow

- Preserve the Experiment / Run / Artifact / Evidence / Claim / Trajectory architecture.
- Keep benchmark-specific behavior in `src/experiment_infra/benchmarks/`.
- Do not add multi-agent orchestration, training, or automatic recovery without a new user request.
- The user requests that every completed version be committed and pushed to `origin/main`. Run relevant checks, review the diff, then commit and push without asking again. Never force-push shared history.
- Keep credentials, benchmark bundles, reference programs, task text and generated benchmark exports out of Git. Respect the upstream data redistribution terms.
- Clearly separate reference-fixture environment calibration from real-model benchmark results.
