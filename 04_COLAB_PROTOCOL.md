# Google Colab Protocol

## Purpose

Use Google Colab GPU for expensive experiments while keeping implementation in the repository.

## Workflow

1. Push the repository to GitHub.
2. Open a Colab notebook.
3. Clone the repository.
4. Mount Google Drive.
5. Select a GPU runtime.
6. Install `requirements.txt`.
7. Verify CUDA/GPU.
8. Run a small smoke test.
9. Launch the repository's training/experiment script with its config.
10. Save checkpoints, logs, metrics, and figures to persistent Drive storage.
11. Sync/copy final artifacts back to the repository.
12. Update `progress.md`.

Example pattern:

```bash
git clone <repo>
cd <repo>
pip install -r requirements.txt
python task2/train.py --config task2/configs/source_only.yaml
```

The exact command must match the implemented repository.

## Persistence

Colab VMs are temporary. A recommended Drive layout is:

```text
MyDrive/
  AToML_PA1/
    checkpoints/
    results/
    figures/
    logs/
```

Do not keep the only copy of important artifacts under `/content`.

## Compute discipline

Smoke tests can be small. Final experiments must follow the assignment protocol. Do not silently reduce epochs, data, interventions, or controlled-study settings.

If a real compute constraint appears, document it before changing the protocol.

## Recovery

Long runs should save checkpoints and metrics periodically and support resuming where practical.

## Talha's role

Talha mainly needs to select/connect the GPU runtime, run prepared commands, monitor failures, maintain Drive persistence, and sync artifacts. The agent should prepare the commands and explain what each run tests.
