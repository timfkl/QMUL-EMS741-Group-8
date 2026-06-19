from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


ROOT = Path(__file__).resolve().parents[1]
LOGDIR = ROOT / "video_demo" / "tensorboard_logs"


def add_image(writer, tag, path, step=0):
    img = np.asarray(Image.open(path).convert("RGB"))
    writer.add_image(tag, img, step, dataformats="HWC")


def main():
    LOGDIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(LOGDIR)

    image_paths = {
        "01_data/few_shot_task_gallery": ROOT / "video_demo" / "assets" / "data_gallery.png",
        "02_model/unet_architecture": ROOT / "video_demo" / "assets" / "unet_architecture.png",
        "03_model/reptile_workflow": ROOT / "video_demo" / "assets" / "reptile_workflow.png",
        "04_episode/support_query_split": ROOT / "video_demo" / "assets" / "few_shot_episode.png",
        "05_results/qualitative_5shot": ROOT / "results" / "qualitative_combined_nshot5.png",
        "06_results/per_structure_dice": ROOT / "results" / "per_structure_results.png",
        "06_results/shot_count_effect": ROOT / "results" / "shot_count_per_structure.png",
        "07_summary/metric_cards": ROOT / "video_demo" / "assets" / "metric_cards.png",
        "08_extension/cnn_architecture": ROOT / "video_demo" / "assets" / "cnn_architecture_extension.png",
        "08_extension/nii_volume_preview": ROOT / "video_demo" / "assets" / "nii_volume_preview.png",
    }
    for tag, path in image_paths.items():
        add_image(writer, tag, path)

    df = pd.read_csv(ROOT / "results" / "per_structure_results_full.csv")
    for _, row in df.iterrows():
        step = int(row["n_shot"])
        writer.add_scalar(f"dice/reptile_{row['task']}", float(row["reptile_mean"]), step)
        writer.add_scalar(f"dice/baseline_{row['task']}", float(row["baseline_mean"]), step)
        writer.add_scalar(f"dice/reptile_advantage_{row['task']}", float(row["reptile_advantage"]), step)

    overall = df.groupby("n_shot")[["reptile_mean", "baseline_mean"]].mean()
    for shot, row in overall.iterrows():
        writer.add_scalar("dice_overall/reptile", float(row["reptile_mean"]), int(shot))
        writer.add_scalar("dice_overall/baseline", float(row["baseline_mean"]), int(shot))
        writer.add_scalar(
            "dice_overall/reptile_advantage",
            float(row["reptile_mean"] - row["baseline_mean"]),
            int(shot),
        )

    writer.add_text(
        "demo/narrative",
        "Reptile meta-trains a compact U-Net initialization across training tasks, "
        "then adapts on 1, 3, or 5 labelled support slices for unseen test structures.",
        0,
    )
    writer.flush()
    writer.close()
    print(f"Wrote TensorBoard logs to {LOGDIR}")


if __name__ == "__main__":
    main()
