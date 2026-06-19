from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "video_demo" / "assets"
OUT.mkdir(parents=True, exist_ok=True)


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def load_gray(path, size=256):
    return np.asarray(Image.open(path).convert("L").resize((size, size)))


def pair_for_mask(mask_path):
    return mask_path.parents[1] / "images" / mask_path.name


def strongest_masks(task_dir, n=1):
    masks = sorted((task_dir / "masks").glob("*.png"))
    scored = []
    for mask in masks:
        arr = np.asarray(Image.open(mask).convert("L"))
        scored.append((int((arr > 127).sum()), mask))
    scored = [item for item in scored if item[0] > 0] or scored
    scored.sort(reverse=True, key=lambda item: item[0])
    if n == 1:
        return [scored[0][1]]
    pick_idx = np.linspace(0, max(0, len(scored) - 1), n, dtype=int)
    return [scored[i][1] for i in pick_idx]


def overlay_mask(image, mask, color=(255, 48, 108), alpha=0.48):
    img = image.astype(np.float32)
    rgb = np.stack([img, img, img], axis=-1)
    mask_bin = mask > 127
    rgb[mask_bin] = (1 - alpha) * rgb[mask_bin] + alpha * np.array(color)
    return np.clip(rgb / 255.0, 0, 1)


def overlay_labels(image, labels, alpha=0.5):
    image = image.astype(np.float32)
    if image.max() > image.min():
        image = (image - image.min()) / (image.max() - image.min())
    rgb = np.stack([image, image, image], axis=-1)
    colors = np.array(
        [
            [0.90, 0.10, 0.20],
            [0.10, 0.45, 0.90],
            [0.10, 0.70, 0.32],
            [0.95, 0.55, 0.12],
            [0.55, 0.25, 0.90],
            [0.00, 0.70, 0.75],
            [0.95, 0.25, 0.60],
            [0.55, 0.75, 0.10],
        ]
    )
    label_int = labels.astype(int)
    for label in sorted(set(np.unique(label_int)) - {0}):
        mask = label_int == label
        color = colors[(label - 1) % len(colors)]
        rgb[mask] = (1 - alpha) * rgb[mask] + alpha * color
    return np.clip(rgb, 0, 1)


def pil_overlay_pair(split, task, size=(210, 210)):
    mask_path = strongest_masks(ROOT / split / task, 1)[0]
    img = load_gray(pair_for_mask(mask_path), size=size[0])
    mask = load_gray(mask_path, size=size[0])
    arr = (overlay_mask(img, mask) * 255).astype(np.uint8)
    return Image.fromarray(arr).resize(size)


def draw_round_rect(draw, xy, fill, outline="#111827", radius=14, width=3):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_center_text(draw, xy, text, fnt, fill="#111827", spacing=6):
    x1, y1, x2, y2 = xy
    lines = text.split("\n")
    heights = []
    widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=fnt)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    total_h = sum(heights) + spacing * (len(lines) - 1)
    y = y1 + (y2 - y1 - total_h) / 2
    for line, w, h in zip(lines, widths, heights):
        draw.text((x1 + (x2 - x1 - w) / 2, y), line, font=fnt, fill=fill)
        y += h + spacing


def arrow(draw, start, end, fill="#111827", width=5):
    draw.line([start, end], fill=fill, width=width)
    sx, sy = start
    ex, ey = end
    angle = np.arctan2(ey - sy, ex - sx)
    head = 18
    left = (ex - head * np.cos(angle - 0.45), ey - head * np.sin(angle - 0.45))
    right = (ex - head * np.cos(angle + 0.45), ey - head * np.sin(angle + 0.45))
    draw.polygon([end, left, right], fill=fill)


def make_ml_data_structure_visuals():
    W, H = 1920, 1080
    title_font = font(56, bold=True)
    h_font = font(34, bold=True)
    body_font = font(25)
    small_font = font(21)

    rows = [
        ("train", ["task_2", "task_3", "task_5", "task_7"], "#dbeafe", "Meta-train"),
        ("val", ["task_4", "task_6"], "#fce7f3", "Tune/checkpoint"),
        ("test", ["task_1", "task_8"], "#dcfce7", "Final evaluation"),
    ]
    stages = [
        "Dataset split folders",
        "Tasks become organs",
        "Image + mask pairs",
        "Few-shot episode",
        "Adapt and evaluate",
    ]
    frames = []

    def base_frame(active_stage=4):
        img = Image.new("RGB", (W, H), "#ffffff")
        draw = ImageDraw.Draw(img)
        draw.text((70, 48), "ML data structure: from files to few-shot episodes", font=title_font, fill="#111827")
        draw.text(
            (72, 118),
            "Talk-over slide: the important idea is task separation, then support/query splitting.",
            font=body_font,
            fill="#4b5563",
        )

        x_positions = [90, 420, 770, 1130, 1510]
        y_top = 210
        for i, label in enumerate(stages):
            fill = "#111827" if i <= active_stage else "#e5e7eb"
            text_fill = "#ffffff" if i <= active_stage else "#6b7280"
            draw_round_rect(draw, (x_positions[i], y_top, x_positions[i] + 260, y_top + 68), fill=fill, outline=fill, radius=10, width=1)
            draw_center_text(draw, (x_positions[i], y_top, x_positions[i] + 260, y_top + 68), label, small_font, fill=text_fill)
            if i < len(stages) - 1:
                arrow(draw, (x_positions[i] + 272, y_top + 34), (x_positions[i + 1] - 14, y_top + 34), fill="#9ca3af", width=4)
        return img, draw, x_positions

    def add_content(img, draw, active_stage=4):
        x_positions = [90, 420, 770, 1130, 1510]
        y0 = 330
        for row_i, (split, tasks, color, role) in enumerate(rows):
            y = y0 + row_i * 190
            draw_round_rect(draw, (x_positions[0], y, x_positions[0] + 260, y + 128), fill=color, radius=12, width=2)
            draw.text((x_positions[0] + 22, y + 24), split + "/", font=h_font, fill="#111827")
            draw.text((x_positions[0] + 22, y + 72), role, font=small_font, fill="#374151")
            if active_stage >= 1:
                task_text = "\n".join(tasks)
                draw_round_rect(draw, (x_positions[1], y, x_positions[1] + 260, y + 128), fill="#f8fafc", radius=12, width=2)
                draw_center_text(draw, (x_positions[1] + 20, y + 12, x_positions[1] + 240, y + 116), task_text, body_font, fill="#111827", spacing=2)
                arrow(draw, (x_positions[0] + 270, y + 64), (x_positions[1] - 16, y + 64), fill="#64748b", width=4)
            if active_stage >= 2:
                thumb = pil_overlay_pair(split, tasks[0], size=(112, 112))
                img.paste(thumb, (x_positions[2] + 18, y + 8))
                draw_round_rect(draw, (x_positions[2] + 145, y + 8, x_positions[2] + 260, y + 120), fill="#f8fafc", radius=10, width=2)
                draw_center_text(draw, (x_positions[2] + 145, y + 8, x_positions[2] + 260, y + 120), "images/\n+\nmasks/", small_font, fill="#111827")
                arrow(draw, (x_positions[1] + 270, y + 64), (x_positions[2] - 16, y + 64), fill="#64748b", width=4)
            if active_stage >= 3:
                draw_round_rect(draw, (x_positions[3], y, x_positions[3] + 260, y + 128), fill="#fff7ed", radius=12, width=2)
                draw.text((x_positions[3] + 22, y + 20), "Support", font=body_font, fill="#1d4ed8")
                draw.text((x_positions[3] + 22, y + 56), "1 / 3 / 5 labelled", font=small_font, fill="#374151")
                draw.text((x_positions[3] + 22, y + 88), "Query: remaining", font=small_font, fill="#be123c")
                arrow(draw, (x_positions[2] + 270, y + 64), (x_positions[3] - 16, y + 64), fill="#64748b", width=4)
            if active_stage >= 4:
                draw_round_rect(draw, (x_positions[4], y, x_positions[4] + 300, y + 128), fill="#ecfdf5", radius=12, width=2)
                draw.text((x_positions[4] + 22, y + 18), "Reptile init", font=body_font, fill="#047857")
                draw.text((x_positions[4] + 22, y + 54), "30-step adaptation", font=small_font, fill="#374151")
                draw.text((x_positions[4] + 22, y + 86), "Dice on query set", font=small_font, fill="#374151")
                arrow(draw, (x_positions[3] + 270, y + 64), (x_positions[4] - 16, y + 64), fill="#64748b", width=4)

        draw_round_rect(draw, (90, 940, 1830, 1015), fill="#f8fafc", outline="#cbd5e1", radius=12, width=2)
        draw.text((120, 962), "Key point for the audience:", font=h_font, fill="#111827")
        draw.text(
            (570, 968),
            "the model never mixes train/val/test tasks; adaptation uses support slices, evaluation uses query slices.",
            font=body_font,
            fill="#334155",
        )
        return img

    static, draw, _ = base_frame(active_stage=4)
    add_content(static, draw, active_stage=4)
    static.save(OUT / "ml_data_structure_map.png")

    for stage in range(5):
        frame, frame_draw, _ = base_frame(active_stage=stage)
        add_content(frame, frame_draw, active_stage=stage)
        frames.extend([frame] * (8 if stage < 4 else 14))
    frames[0].save(
        OUT / "ml_data_structure.gif",
        save_all=True,
        append_images=frames[1:],
        duration=140,
        loop=0,
        optimize=False,
    )


def make_data_gallery():
    tasks = [
        ("train", "task_2", "Meta-train"),
        ("train", "task_3", "Meta-train"),
        ("train", "task_5", "Meta-train"),
        ("train", "task_7", "Meta-train"),
        ("val", "task_4", "Validation"),
        ("val", "task_6", "Validation"),
        ("test", "task_1", "Held-out test"),
        ("test", "task_8", "Held-out test"),
    ]
    fig, axes = plt.subplots(4, 4, figsize=(11.0, 9.6), dpi=180)
    for idx, (split, task, role) in enumerate(tasks):
        row = idx // 2
        col_offset = (idx % 2) * 2
        mask_path = strongest_masks(ROOT / split / task, 1)[0]
        img = load_gray(pair_for_mask(mask_path))
        mask = load_gray(mask_path)
        axes[row, col_offset].imshow(img, cmap="gray")
        axes[row, col_offset + 1].imshow(overlay_mask(img, mask))
        for ax in (axes[row, col_offset], axes[row, col_offset + 1]):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[row, col_offset].set_ylabel(
            f"{role}\n{split}/{task}",
            rotation=0,
            ha="right",
            va="center",
            fontsize=9,
            labelpad=20,
        )
    axes[0, 0].set_title("MRI slice", fontsize=11, weight="bold")
    axes[0, 1].set_title("Mask overlay", fontsize=11, weight="bold")
    axes[0, 2].set_title("MRI slice", fontsize=11, weight="bold")
    axes[0, 3].set_title("Mask overlay", fontsize=11, weight="bold")
    fig.suptitle("Few-shot segmentation tasks", fontsize=16, weight="bold", y=0.98)
    fig.tight_layout(rect=[0.05, 0, 1, 0.95], w_pad=2.0, h_pad=1.0)
    fig.savefig(OUT / "data_gallery.png", facecolor="white")
    plt.close(fig)


def make_episode_visual():
    task_dir = ROOT / "test" / "task_1"
    support_masks = strongest_masks(task_dir, 5)
    query_masks = strongest_masks(task_dir, 3)[::-1]
    fig = plt.figure(figsize=(12, 5.2), dpi=180)
    gs = fig.add_gridspec(2, 7, width_ratios=[1, 1, 1, 1, 1, 0.35, 1.3])

    for i, mask_path in enumerate(support_masks):
        ax = fig.add_subplot(gs[0, i])
        img = load_gray(pair_for_mask(mask_path))
        mask = load_gray(mask_path)
        ax.imshow(overlay_mask(img, mask, color=(54, 162, 235), alpha=0.42))
        ax.set_title(f"shot {i + 1}", fontsize=8)
        ax.axis("off")

    arrow_ax = fig.add_subplot(gs[:, 5])
    arrow_ax.axis("off")
    arrow_ax.add_patch(
        FancyArrowPatch(
            (0.08, 0.5),
            (0.95, 0.5),
            arrowstyle="-|>",
            mutation_scale=28,
            linewidth=2.4,
            color="#111827",
        )
    )

    for i, mask_path in enumerate(query_masks):
        ax = fig.add_subplot(gs[1, i])
        img = load_gray(pair_for_mask(mask_path))
        mask = load_gray(mask_path)
        ax.imshow(overlay_mask(img, mask, color=(255, 48, 108), alpha=0.46))
        ax.set_title(f"query {i + 1}", fontsize=8)
        ax.axis("off")

    text_ax = fig.add_subplot(gs[:, 6])
    text_ax.axis("off")
    text_ax.text(
        0.02,
        0.72,
        "5 annotated support slices\nadapt the meta-initialized U-Net",
        fontsize=12,
        weight="bold",
        ha="left",
        va="center",
    )
    text_ax.text(
        0.02,
        0.42,
        "Remaining query slices are\nheld back for Dice evaluation",
        fontsize=10,
        color="#374151",
        ha="left",
        va="center",
    )
    fig.suptitle("Few-shot episode on held-out task_1", fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "few_shot_episode.png", facecolor="white")
    plt.close(fig)


def make_reptile_workflow():
    fig, ax = plt.subplots(figsize=(12.6, 5.8), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.04, 0.62, 0.17, 0.18, "Sample task", "train task 2/3/5/7"),
        (0.30, 0.62, 0.17, 0.18, "Clone U-Net", "start from theta"),
        (0.56, 0.62, 0.17, 0.18, "Inner loop", "10 SGD steps"),
        (0.79, 0.62, 0.17, 0.18, "Meta update", "theta moves toward fast weights"),
        (0.30, 0.18, 0.17, 0.18, "Validate", "5-shot val adaptation"),
        (0.56, 0.18, 0.17, 0.18, "Checkpoint", "best validation Dice"),
    ]
    colors = ["#dbeafe", "#ede9fe", "#dcfce7", "#fff7ed", "#fce7f3", "#e0f2fe"]
    for (x, y, w, h, title, body), color in zip(boxes, colors):
        ax.add_patch(Rectangle((x, y), w, h, linewidth=1.6, edgecolor="#111827", facecolor=color))
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=12, weight="bold")
        ax.text(x + w / 2, y + h * 0.35, body, ha="center", va="center", fontsize=9, color="#374151")

    arrows = [
        ((0.21, 0.71), (0.30, 0.71)),
        ((0.47, 0.71), (0.56, 0.71)),
        ((0.73, 0.71), (0.79, 0.71)),
        ((0.875, 0.62), (0.875, 0.46)),
        ((0.875, 0.46), (0.13, 0.46)),
        ((0.13, 0.46), (0.13, 0.62)),
        ((0.385, 0.62), (0.385, 0.36)),
        ((0.47, 0.27), (0.56, 0.27)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, linewidth=1.8, color="#111827"))

    ax.text(
        0.5,
        0.92,
        "Reptile learns an initialization that adapts quickly",
        ha="center",
        va="center",
        fontsize=19,
        weight="bold",
    )
    ax.text(
        0.5,
        0.05,
        "theta <- theta + alpha * (fast weights - theta), with alpha decayed over 8000 outer steps",
        ha="center",
        va="center",
        fontsize=11,
        color="#374151",
    )
    fig.tight_layout()
    fig.savefig(OUT / "reptile_workflow.png", facecolor="white")
    plt.close(fig)


def make_unet_architecture():
    fig, ax = plt.subplots(figsize=(13.2, 5.0), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    levels = [
        (0.06, 0.70, "32", "256x256"),
        (0.18, 0.58, "64", "128x128"),
        (0.30, 0.46, "128", "64x64"),
        (0.42, 0.34, "256", "32x32"),
        (0.54, 0.22, "512", "16x16"),
        (0.66, 0.34, "256", "32x32"),
        (0.78, 0.46, "128", "64x64"),
        (0.88, 0.58, "64", "128x128"),
        (0.96, 0.70, "1", "256x256"),
    ]
    for i, (x, y, ch, size) in enumerate(levels):
        w = 0.065 if i not in (0, 8) else 0.05
        h = 0.18 + 0.018 * (4 - abs(4 - i))
        color = "#bfdbfe" if i < 4 else "#bbf7d0" if i > 4 else "#fde68a"
        ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor=color, edgecolor="#111827", linewidth=1.4))
        ax.text(x, y + 0.025, f"{ch} ch", ha="center", va="center", fontsize=9, weight="bold")
        ax.text(x, y - 0.045, size, ha="center", va="center", fontsize=7, color="#374151")
    for i in range(len(levels) - 1):
        ax.add_patch(FancyArrowPatch((levels[i][0] + 0.035, levels[i][1]), (levels[i + 1][0] - 0.035, levels[i + 1][1]), arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color="#111827"))
    for left, right in [(0, 8), (1, 7), (2, 6), (3, 5)]:
        ax.plot([levels[left][0], levels[right][0]], [levels[left][1] + 0.17, levels[right][1] + 0.17], color="#e11d48", linewidth=1.6)
        ax.text((levels[left][0] + levels[right][0]) / 2, levels[left][1] + 0.19, "skip", ha="center", fontsize=8, color="#9f1239")
    ax.text(0.5, 0.94, "Compact U-Net backbone", ha="center", va="center", fontsize=19, weight="bold")
    ax.text(0.5, 0.08, "1-channel MRI slice -> sigmoid probability mask; DoubleConv blocks use GroupNorm for small batches", ha="center", fontsize=10, color="#374151")
    fig.tight_layout()
    fig.savefig(OUT / "unet_architecture.png", facecolor="white")
    plt.close(fig)


def make_metric_cards():
    df = pd.read_csv(ROOT / "results" / "per_structure_results_full.csv")
    overall = df.groupby("n_shot")[["reptile_mean", "baseline_mean"]].mean().reset_index()
    overall["advantage"] = overall["reptile_mean"] - overall["baseline_mean"]

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=180)
    ax.axis("off")
    ax.text(0.5, 0.92, "Reptile improves few-shot Dice across shot counts", ha="center", fontsize=17, weight="bold")
    x0 = 0.08
    for i, row in overall.iterrows():
        x = x0 + i * 0.31
        ax.add_patch(Rectangle((x, 0.28), 0.24, 0.44, facecolor="#f8fafc", edgecolor="#111827", linewidth=1.2))
        ax.text(x + 0.12, 0.63, f"{int(row.n_shot)}-shot", ha="center", fontsize=13, weight="bold")
        ax.text(x + 0.12, 0.50, f"Reptile {row.reptile_mean:.3f}", ha="center", fontsize=11, color="#1d4ed8")
        ax.text(x + 0.12, 0.40, f"Baseline {row.baseline_mean:.3f}", ha="center", fontsize=11, color="#c2410c")
        ax.text(x + 0.12, 0.30, f"+{row.advantage:.3f} Dice", ha="center", fontsize=12, weight="bold", color="#047857")
    ax.text(0.5, 0.12, "Averages are computed from results/per_structure_results_full.csv over held-out task_1 and task_8.", ha="center", fontsize=9, color="#475569")
    fig.tight_layout()
    fig.savefig(OUT / "metric_cards.png", facecolor="white")
    plt.close(fig)


def make_nii_volume_preview():
    img_path = ROOT / "001000_img.nii"
    mask_path = ROOT / "001000_mask.nii"
    used_real_nii = img_path.exists() and mask_path.exists()

    if used_real_nii:
        import nibabel as nib

        volume = nib.load(img_path).get_fdata().astype(np.float32)
        labels = nib.load(mask_path).get_fdata().astype(np.int16)
        low, high = np.percentile(volume, [1, 99])
        volume = np.clip((volume - low) / max(high - low, 1e-6), 0, 1)
        x_idx = int(np.argmax(labels.sum(axis=(1, 2))))
        y_idx = int(np.argmax(labels.sum(axis=(0, 2))))
        z_idx = int(np.argmax(labels.sum(axis=(0, 1))))

        views = [
            ("Axial + label map", np.rot90(volume[:, :, z_idx]), np.rot90(labels[:, :, z_idx])),
            ("Coronal + label map", np.rot90(volume[:, y_idx, :]), np.rot90(labels[:, y_idx, :])),
            ("Sagittal + label map", np.rot90(volume[x_idx, :, :]), np.rot90(labels[x_idx, :, :])),
        ]
        source_note = "Source files: 001000_img.nii and 001000_mask.nii"
        volume_title = "Real 3D NIfTI scan with multi-label mask"
    else:
        image_dir = ROOT / "test" / "task_1" / "images"
        grouped = {}
        for path in sorted(image_dir.glob("*.png")):
            if "_slice_" not in path.stem:
                continue
            sample_id = path.stem.split("_slice_")[0]
            grouped.setdefault(sample_id, []).append(path)
        sample_id, paths = max(grouped.items(), key=lambda item: len(item[1]))
        volume = np.stack([load_gray(path, size=192) for path in sorted(paths)], axis=0)
        volume = volume.astype(np.float32)
        low, high = np.percentile(volume, [1, 99])
        volume = np.clip((volume - low) / max(high - low, 1e-6), 0, 1)
        labels = np.zeros_like(volume, dtype=np.int16)
        z_idx = volume.shape[0] // 2
        y_idx = volume.shape[1] // 2
        x_idx = volume.shape[2] // 2
        views = [
            ("Axial slice", volume[z_idx], labels[z_idx]),
            ("Coronal reconstruction", volume[:, y_idx, :], labels[:, y_idx, :]),
            ("Sagittal reconstruction", volume[:, :, x_idx], labels[:, :, x_idx]),
        ]
        source_note = f"Fallback preview from PNG stack: test/task_1/images/{sample_id}_slice_*.png"
        volume_title = "NIfTI-style volume preview from slice stack"

    fig = plt.figure(figsize=(13.6, 7.65), dpi=180)
    fig.text(
        0.5,
        0.94,
        "Conference extension preview: 3D pelvic MRI segmentation",
        ha="center",
        va="center",
        fontsize=20,
        weight="bold",
    )
    image_positions = [
        (0.045, 0.46, 0.205, 0.33),
        (0.285, 0.46, 0.205, 0.33),
        (0.525, 0.46, 0.205, 0.33),
    ]
    for i, (title, arr, label_arr) in enumerate(views):
        ax = fig.add_axes(image_positions[i])
        ax.imshow(overlay_labels(arr, label_arr, alpha=0.55), aspect="auto")
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    stack_ax = fig.add_axes((0.77, 0.34, 0.20, 0.46))
    stack_ax.set_title("Stacked-slice volume cue", fontsize=11, weight="bold")
    stack_ax.axis("off")
    if used_real_nii:
        slice_axis = 2
        indices = np.linspace(0, volume.shape[2] - 1, 9, dtype=int)
    else:
        slice_axis = 0
        indices = np.linspace(0, volume.shape[0] - 1, 9, dtype=int)
    for offset, idx in enumerate(indices):
        dx = 0.025 * offset
        dy = 0.018 * offset
        if slice_axis == 2:
            stack_img = overlay_labels(np.rot90(volume[:, :, idx]), np.rot90(labels[:, :, idx]), alpha=0.55)
        else:
            stack_img = overlay_labels(volume[idx], labels[idx], alpha=0.55)
        stack_ax.imshow(
            stack_img,
            extent=(0.04 + dx, 0.82 + dx, 0.05 + dy, 0.83 + dy),
            alpha=0.72,
            zorder=offset,
        )
        stack_ax.add_patch(
            Rectangle(
                (0.04 + dx, 0.05 + dy),
                0.78,
                0.78,
                fill=False,
                edgecolor="#111827",
                linewidth=0.7,
                alpha=0.35,
                zorder=offset + 0.2,
            )
        )
    stack_ax.set_xlim(0, 1.08)
    stack_ax.set_ylim(0, 1.02)

    text_ax = fig.add_axes((0.045, 0.08, 0.90, 0.24))
    text_ax.axis("off")
    text_ax.text(
        0.0,
        0.78,
        "Preview of the extension: from 2D slices to 3D scan segmentation",
        fontsize=18,
        weight="bold",
        ha="left",
    )
    text_ax.text(
        0.0,
        0.50,
        "The larger-dataset extension works with NIfTI (.nii) volumes, keeping neighboring slices together "
        "so a 3D CNN can use spatial context through the scan.",
        fontsize=11,
        color="#374151",
        ha="left",
        wrap=True,
    )
    text_ax.text(
        0.0,
        0.23,
        f"{volume_title}. {source_note}",
        fontsize=9,
        color="#64748b",
        ha="left",
    )
    fig.savefig(OUT / "nii_volume_preview.png", facecolor="white")
    plt.close(fig)


def main():
    make_data_gallery()
    make_ml_data_structure_visuals()
    make_episode_visual()
    make_reptile_workflow()
    make_unet_architecture()
    make_metric_cards()
    make_nii_volume_preview()
    print(f"Wrote demo assets to {OUT}")


if __name__ == "__main__":
    main()
