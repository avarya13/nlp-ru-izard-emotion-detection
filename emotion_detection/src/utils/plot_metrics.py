from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_all_plots(
    csv_path: Path, output_dir: Path, model_name: str, timestamp: str
) -> None:
    output_dir = output_dir / model_name.replace("/", "-") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Plot directory: {output_dir}")

    df = pd.read_csv(csv_path)

    metrics = [
        ("train_loss", "val_loss"),
        ("train_f1_macro", "val_f1_macro"),
        ("train_f1_micro", "val_f1_micro"),
        ("train_precision_macro", "val_precision_macro"),
        ("train_recall_macro", "val_recall_macro"),
        ("custom_train_f1_macro", "custom_val_f1_macro"),
        ("custom_train_f1_micro", "custom_val_f1_micro"),
    ]

    for train_metric, val_metric in metrics:
        if train_metric in df.columns and val_metric in df.columns:
            plt.figure()
            train_data = df[df[train_metric].notna()]
            val_data = df[df[val_metric].notna()]
            if not train_data.empty:
                plt.plot(
                    train_data["epoch"], train_data[train_metric], label=train_metric
                )
            if not val_data.empty:
                plt.plot(val_data["epoch"], val_data[val_metric], label=val_metric)
            plt.xlabel("Epoch")
            plt.ylabel("Value")
            plt.title(f"{train_metric} / {val_metric}")
            plt.legend()
            plt.grid(True)
            plt.savefig(
                output_dir / f"{train_metric}_{val_metric}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

    if "train_loss" in df.columns and "val_loss" in df.columns:
        plt.figure()
        train_loss = df[df["train_loss"].notna()]
        val_loss = df[df["val_loss"].notna()]
        plt.plot(train_loss["epoch"], train_loss["train_loss"], label="train_loss")
        plt.plot(val_loss["epoch"], val_loss["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
        plt.close()
