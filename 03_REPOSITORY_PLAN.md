# Repository Plan

The agent must implement the repository, not merely describe it.

Minimum structure:

```text
README.md
AGENT.md
progress.md
DIRECTORY_STRUCTURE.md
COLAB.md
requirements.txt
.gitignore

common/
shared/
task1/
task2/
task3/
task4/
report/
figures/
```

Each task should have clear locations for configs, source code, data manifests, checkpoints, logs, metrics, figures, and analysis artifacts.

Prefer configuration-driven execution.

The repository should expose documented Python entry points for Tasks 1–4. Colab should execute these scripts rather than duplicate implementation in notebook cells.
