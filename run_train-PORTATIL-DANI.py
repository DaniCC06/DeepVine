from __future__ import annotations

from train import TrainingConfig, run_training


def main() -> None:
    cfg = TrainingConfig(
    dataset_dir=r"D:\dataset",
    resume_from_checkpoint=False,
)
    history = run_training(cfg)

    print("Training finished.")
    print(f"Best macro F1: {history.best_macro_f1:.4f}")
    print(f"Best checkpoint: {history.best_checkpoint_path}")


if __name__ == "__main__":
    main()
