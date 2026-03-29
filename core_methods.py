"""
Core methods for EMS741 few-shot segmentation.

Model, data utilities, loss/metrics, Reptile meta-training, and unified
adapt-and-evaluate functions so notebooks stay clean.
"""

from pathlib import Path
import copy
import random

from datetime import datetime
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import CosineAnnealingLR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256

# ---------------------------------------------------------------------
# Global config and utilities
# ---------------------------------------------------------------------

BCE_WEIGHT = 0.2  # default BCE weight in BCE+Dice
RUN_ID = datetime.now().strftime("%y%m%d_%H%M")

DEFAULT_CONFIG = {
    "N_OUTER": 2000,
    "K_INNER": 30,
    "META_LR": 0.1,
    "META_LR_MIN_FACTOR": 0.05,  # floor at 5% of initial meta LR
    "INNER_LR": 1e-3,
    "VAL_EVERY": 200,
    "N_SHOT_VAL": 5,
    "ADAPT_STEPS": 30,
    "ADAPT_LR": 1e-3,
    "BASELINE_EPOCHS": 30,
    "BASELINE_LR": 1e-3,
    "BCE_WEIGHT": BCE_WEIGHT,
    # Optional extras (disabled by default)
    "NORM": "group",  # 'batch' or 'group'
    "USE_VAL_DICE_EMA": True,
    "VAL_DICE_EMA_ALPHA": 0.3,
    "CHECKPOINT_BEST": True,
    "CHECKPOINT_PATH": f"metamodel_best-{RUN_ID}.pt",
}


def set_seed(seed: int) -> None:
    """Seed torch, numpy, and random for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------


def discover_tasks(split_dir: Path):
    """
    Return dict {task_name: {'images': [...], 'masks': [...]}}.

    Expects split_dir/taskX/images/*.png and split_dir/taskX/masks/*.png.
    """
    split_dir = Path(split_dir)
    tasks = {}
    for task_dir in sorted(split_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        img_dir = task_dir / "images"
        mask_dir = task_dir / "masks"
        imgs = sorted(img_dir.glob("*.png"))
        masks = sorted(mask_dir.glob("*.png"))
        assert len(imgs) == len(masks), f"{task_dir} img/mask count mismatch"
        tasks[task_dir.name] = {"images": imgs, "masks": masks}
    return tasks


def load_sample(img_path, mask_path, img_size: int = IMG_SIZE):
    """
    Load one image-mask pair.

    Returns
    -------
    image : np.ndarray float32 in [0, 1], shape (H, W)
    mask : np.ndarray uint8 in {0, 1}, shape (H, W)
    """
    img = Image.open(img_path).convert("L").resize(
        (img_size, img_size), Image.BILINEAR
    )
    mask = Image.open(mask_path).convert("L").resize(
        (img_size, img_size), Image.NEAREST
    )

    img_arr = np.asarray(img, dtype=np.float32) / 255.0
    mask_arr = np.asarray(mask, dtype=np.uint8)

    if mask_arr.max() > 1:
        mask_arr = (mask_arr > 127).astype(np.uint8)

    return img_arr, mask_arr


class SegDataset(Dataset):
    """Simple single-task segmentation dataset."""

    def __init__(self, task_dict, augment: bool = False):
        self.images = task_dict["images"]
        self.masks = task_dict["masks"]
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, mask = load_sample(self.images[idx], self.masks[idx], IMG_SIZE)

        img_t = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        if self.augment:
            if random.random() > 0.5:
                img_t = TF.hflip(img_t)
                mask_t = TF.hflip(mask_t)
            if random.random() > 0.5:
                img_t = TF.vflip(img_t)
                mask_t = TF.vflip(mask_t)
            angle = random.uniform(-15, 15)
            img_t = TF.rotate(img_t, angle)
            mask_t = TF.rotate(mask_t, angle)

        return img_t, mask_t


class FewShotEpisodeDataset:
    """
    Splits a task into n_shot support samples and all remaining slices as query.
    The split is deterministic given the same seed, ensuring reproducible episodes.
    """

    def __init__(self, task_dict, n_shot: int, seed: int = 42):
        rng = random.Random(seed)
        all_idx = list(range(len(task_dict["images"])))
        support_idx = rng.sample(all_idx, n_shot)
        query_idx = [i for i in all_idx if i not in support_idx]

        self.support = {
            "images": [task_dict["images"][i] for i in support_idx],
            "masks":  [task_dict["masks"][i]  for i in support_idx],
        }
        self.query = {
            "images": [task_dict["images"][i] for i in query_idx],
            "masks":  [task_dict["masks"][i]  for i in query_idx],
        }

    def support_loader(self, batch_size=None):
        ds = SegDataset(self.support, augment=False)
        bs = batch_size or len(ds)
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def query_loader(self, batch_size: int = 4):
        ds = SegDataset(self.query, augment=False)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

# Backwards-compatible alias if you want to use the shorter name
FewShotEpisode = FewShotEpisodeDataset

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------


class DoubleConv(nn.Module):
    """
    Two sequential Conv2d → Norm → ReLU blocks.
    Supports 'batch' (BatchNorm2d) or 'group' (GroupNorm, 8 groups) normalisation.
    """
    def __init__(self, in_ch, out_ch, norm: str = "batch"):
        super().__init__()

        def make_norm(ch: int) -> nn.Module:
            if norm == "group":
                # 8 groups works well for channels >= 32
                return nn.GroupNorm(num_groups=8, num_channels=ch)
            else:
                return nn.BatchNorm2d(ch)

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            make_norm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            make_norm(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Compact U-Net: 1-channel in, 1-channel sigmoid mask out."""

    def __init__(self, channels=(32, 64, 128, 256), norm: str = "batch"):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        self.norm = norm

        in_ch = 1
        for ch in channels:
            self.downs.append(DoubleConv(in_ch, ch, norm=norm))
            in_ch = ch

        self.bottleneck = DoubleConv(channels[-1], channels[-1] * 2, norm=norm)

        rev = list(reversed(channels))
        for ch in rev:
            self.ups.append(nn.ConvTranspose2d(ch * 2, ch, 2, stride=2))
            self.ups.append(DoubleConv(ch * 2, ch, norm=norm))

        self.out_conv = nn.Conv2d(channels[0], 1, 1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skip_connections[i // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        return torch.sigmoid(self.out_conv(x))


# ---------------------------------------------------------------------
# Losses and metrics
# ---------------------------------------------------------------------


def dice_loss(pred, target, smooth: float = 1.0):
    pred = pred.view(-1)
    target = target.view(-1)
    inter = (pred * target).sum()
    return 1.0 - (2.0 * inter + smooth) / (pred.sum() + target.sum() + smooth)


def bce_dice_loss(pred, target, bce_weight: float = BCE_WEIGHT):
    bce = F.binary_cross_entropy(pred, target)
    d = dice_loss(pred, target)
    return bce_weight * bce + (1.0 - bce_weight) * d


@torch.no_grad()
def dice_score(pred, target, threshold: float = 0.2, smooth: float = 1.0):
    """
    Compute binary Dice score after thresholding sigmoid predictions.

    threshold: binarisation cutoff. Set to 0.2 (rather than 0.5) to be
               more permissive with low-confidence predictions on small 
               foreground regions typical in abdominal MRI segmentation.
    smooth: Laplace smoothing term to avoid division by zero.
    """
    pred_bin = (pred > threshold).float()
    inter = (pred_bin * target).sum()
    return float(
        (2.0 * inter + smooth) / (pred_bin.sum() + target.sum() + smooth)
    )


# ---------------------------------------------------------------------
# Shared inner-loop helper
# ---------------------------------------------------------------------


def run_inner_loop(
    model: nn.Module,
    support_loader: DataLoader,
    n_steps: int,
    lr: float,
    optimizer_cls=torch.optim.Adam,
    use_scheduler: bool = False,
) -> nn.Module:
    """
    Generic inner loop used by Reptile, test-time adaptation, and baseline.

    Runs exactly n_steps gradient steps on the support set, cycling back to
    the start of support_loader if exhausted (important for small n_shot where
    the loader may have fewer batches than n_steps).

    optimizer_cls: any torch.optim class, passed as a constructor (not instance).
    use_scheduler: if True, applies CosineAnnealingLR over n_steps.
    """
    model.train()
    opt = optimizer_cls(model.parameters(), lr=lr)
    scheduler = (
        CosineAnnealingLR(opt, T_max=n_steps) if use_scheduler else None
    )

    data_iter = iter(support_loader)

    for step in range(n_steps):
        try:
            imgs, masks = next(data_iter)
        except StopIteration:
            data_iter = iter(support_loader)
            imgs, masks = next(data_iter)

        imgs = imgs.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)
        opt.zero_grad()
        preds = model(imgs)
        loss = bce_dice_loss(preds, masks)
        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()

    return model


# ---------------------------------------------------------------------
# Baseline training for a single task (standalone)
# ---------------------------------------------------------------------


def train_baseline(
    task_dict,
    n_shot: int,
    n_epochs: int = 20,  # kept for backwards-compat, experiment uses unified path
    lr: float = 1e-3,
    seed: int = 42,
):
    """
    Train a fresh UNet on n_shot examples of a task.

    .. deprecated::
        This function is kept for backwards compatibility. All experiments
        now use unified_adapt_and_evaluate(), which shares the inner loop
        with adapt_and_evaluate for consistency.

    Returns (history, model, episode).
    """
    set_seed(seed)

    episode = FewShotEpisodeDataset(task_dict, n_shot, seed=seed)
    support_loader = episode.support_loader(batch_size=min(n_shot, 4))
    query_loader = episode.query_loader(batch_size=4)

    model = UNet(norm=DEFAULT_CONFIG["NORM"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_dice": []}

    for _ in range(n_epochs):
        model.train()
        running_loss = 0.0
        for imgs, masks in support_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            preds = model(imgs)
            loss = bce_dice_loss(preds, masks)
            loss.backward()
            opt.step()
            running_loss += loss.item()

        model.eval()
        dice_vals = []
        with torch.no_grad():
            for imgs, masks in query_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)
                dice_vals.append(dice_score(model(imgs), masks))

        history["train_loss"].append(
            running_loss / max(1, len(support_loader))
        )
        history["val_dice"].append(float(np.mean(dice_vals)))

    return history, model, episode


# ---------------------------------------------------------------------
# Reptile meta-training and few-shot evaluation
# ---------------------------------------------------------------------


def reptile_meta_train(
    train_tasks,
    val_tasks,
    n_outer: int = DEFAULT_CONFIG["N_OUTER"],
    k_inner: int = DEFAULT_CONFIG["K_INNER"],
    inner_lr: float = DEFAULT_CONFIG["INNER_LR"],
    meta_lr: float = DEFAULT_CONFIG["META_LR"],
    batch_size: int = 4,
    val_every: int = DEFAULT_CONFIG["VAL_EVERY"],
    n_shot_val: int = DEFAULT_CONFIG["N_SHOT_VAL"],
    meta_lr_min_factor: float = DEFAULT_CONFIG["META_LR_MIN_FACTOR"],
    use_val_ema: bool = DEFAULT_CONFIG["USE_VAL_DICE_EMA"],
    ema_alpha: float = DEFAULT_CONFIG["VAL_DICE_EMA_ALPHA"],
    checkpoint_best: bool = DEFAULT_CONFIG["CHECKPOINT_BEST"],
    checkpoint_path: str = DEFAULT_CONFIG["CHECKPOINT_PATH"],
):
    """
    Reptile meta-training loop with val logging and decaying meta-LR.

    Returns (meta_model, history) where history has:
    - "outer_step": list of steps where validation was run
    - "val_dice":   validation Dice at each val step (optionally EMA-smoothed)
    - "meta_loss":  placeholder zeros (inner loss not tracked for performance)
    - "meta_lr":    meta learning rate at each outer step
        """
    meta_model = UNet(norm=DEFAULT_CONFIG["NORM"]).to(DEVICE)
    task_names = list(train_tasks.keys())
    history = {"outer_step": [], "val_dice": [], "meta_loss": [], "meta_lr": []}

    best_val = -1.0
    best_weights = None
    best_step = 0

    print(
        f"Reptile meta-training: n_outer={n_outer}, "
        f"k_inner={k_inner}, meta_lr={meta_lr}"
    )

    for outer in range(1, n_outer + 1):
        # Sample a random training task
        task_name = random.choice(task_names)
        task_dict = train_tasks[task_name]
        dataset = SegDataset(task_dict, augment=True)
        support_loader = DataLoader(
            dataset, 
            batch_size=min(batch_size, len(dataset)), 
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

        # Clone meta-parameters for inner loop
        fast = copy.deepcopy(meta_model)

        # Inner loop using shared helper (SGD, no scheduler)
        fast = run_inner_loop(
            fast,
            support_loader,
            n_steps=k_inner,
            lr=inner_lr,
            optimizer_cls=torch.optim.SGD,
            use_scheduler=False,
        )

        # Compute current meta-LR with linear decay
        progress = outer / float(n_outer)
        meta_lr_t = meta_lr * (1.0 - progress)
        if meta_lr_min_factor is not None:
            meta_lr_t = max(meta_lr_t, meta_lr * meta_lr_min_factor)

        # Reptile meta-update
        with torch.no_grad():
            for mp, fp in zip(meta_model.parameters(), fast.parameters()):
                # meta_p += meta_lr_t * (fast_p - meta_p)
                mp.data += meta_lr_t * (fp.data - mp.data)

        # For simplicity, store meta_lr_t and dummy inner loss (0.0)
        history["meta_lr"].append(meta_lr_t)
        history["meta_loss"].append(0.0)

        if outer % 50 == 0:
            print(
                f"[Reptile] outer {outer:5d}/{n_outer} "
                f"meta_lr_t {meta_lr_t:.4f}"
            )

        if outer % val_every == 0:
            val_d = evaluate_few_shot(
                meta_model,
                val_tasks,
                n_shot=n_shot_val,
                adapt_steps=DEFAULT_CONFIG["ADAPT_STEPS"],
                adapt_lr=DEFAULT_CONFIG["ADAPT_LR"],
            )

            if use_val_ema and history["val_dice"]:
                prev = history["val_dice"][-1]
                val_smoothed = ema_alpha * val_d + (1.0 - ema_alpha) * prev
            else:
                val_smoothed = val_d

            history["outer_step"].append(outer)
            history["val_dice"].append(val_smoothed)
            print(
                f"[Reptile] outer {outer:5d}/{n_outer} "
                f"val Dice (n_shot={n_shot_val}) {val_d:.4f} "
                f"(smoothed {val_smoothed:.4f})"
            )

            if checkpoint_best and val_d > best_val:
                best_val = val_d
                best_step = outer
                best_weights = copy.deepcopy(meta_model.state_dict())

                torch.save(meta_model.state_dict(), checkpoint_path)
                print(
                    f"  New best val Dice {val_d:.4f}, at step {best_step}, "
                    f"checkpoint saved to {checkpoint_path}"
                )

    return meta_model, history, best_val, best_step, best_weights, checkpoint_path


def adapt_and_evaluate(
    meta_model,
    task_dict,
    n_shot: int,
    adapt_steps: int = DEFAULT_CONFIG["ADAPT_STEPS"],
    adapt_lr: float = DEFAULT_CONFIG["ADAPT_LR"],
    seed: int = 42,
):
    """
    Fine-tune meta_model on n_shot support and evaluate on query set.

    Returns (mean_dice, adapted_model, episode).
    """
    set_seed(seed)

    episode = FewShotEpisodeDataset(task_dict, n_shot, seed=seed)
    adapted = copy.deepcopy(meta_model).to(DEVICE)

    support_loader = episode.support_loader(batch_size=min(n_shot, 4))

    # Shared inner loop (Adam + cosine scheduler)
    adapted = run_inner_loop(
        adapted,
        support_loader,
        n_steps=adapt_steps,
        lr=adapt_lr,
        optimizer_cls=torch.optim.Adam,
        use_scheduler=True,
    )

    adapted.eval()
    dice_vals = []
    with torch.no_grad():
        for imgs, masks in episode.query_loader(batch_size=4):
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            dice_vals.append(dice_score(adapted(imgs), masks))

    return float(np.mean(dice_vals)), adapted, episode


def evaluate_few_shot(
    meta_model,
    tasks,
    n_shot: int,
    adapt_steps: int = DEFAULT_CONFIG["ADAPT_STEPS"],
    adapt_lr: float = DEFAULT_CONFIG["ADAPT_LR"],
):
    """
    Average adapt_and_evaluate Dice over all tasks in a split.
    Used internally by reptile_meta_train for validation.

    Returns mean Dice (float) across all tasks.
    """
    scores = []
    for task_dict in tasks.values():
        d, _, _ = adapt_and_evaluate(
            meta_model,
            task_dict,
            n_shot=n_shot,
            adapt_steps=adapt_steps,
            adapt_lr=adapt_lr,
        )
        scores.append(d)
    return float(np.mean(scores))


def unified_adapt_and_evaluate(
    base_model,
    task_dict,
    n_shot: int,
    epochs: int = DEFAULT_CONFIG["BASELINE_EPOCHS"],
    lr: float = DEFAULT_CONFIG["BASELINE_LR"],
    seed: int = 42,
    use_scheduler: bool = True,
    optimizer_cls=torch.optim.Adam,
):
    """
    Generic adaptation wrapper used for baselines and meta-inits.

    Uses the same style of inner loop as adapt_and_evaluate but with epochs.
    """
    set_seed(seed)

    episode = FewShotEpisodeDataset(task_dict, n_shot, seed=seed)
    adapted = copy.deepcopy(base_model).to(DEVICE)
    support_loader = episode.support_loader(batch_size=min(n_shot, 4))

    adapted = run_inner_loop(
        adapted,
        support_loader,
        n_steps=epochs,
        lr=lr,
        optimizer_cls=optimizer_cls,
        use_scheduler=use_scheduler,
    )

    adapted.eval()
    dice_vals = []
    with torch.no_grad():
        for imgs, masks in episode.query_loader(batch_size=4):
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            dice_vals.append(dice_score(adapted(imgs), masks))

    return float(np.mean(dice_vals)), adapted, episode