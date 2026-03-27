"""
Core methods for EMS741 few-shot segmentation.

Model, data utilities, loss/metrics, Reptile meta-training, and unified
adapt-and-evaluate functions so notebooks stay clean.
"""

from pathlib import Path
import copy
import random

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256


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
    mask  : np.ndarray uint8  in {0, 1}, shape (H, W)
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
    """Returns exactly n_shot support samples and all remaining as query."""

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
        return DataLoader(ds, batch_size=bs, shuffle=True)

    def query_loader(self, batch_size: int = 4):
        ds = SegDataset(self.query, augment=False)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)


# Backwards-compatible alias if you want to use the shorter name
FewShotEpisode = FewShotEpisodeDataset


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Compact U-Net: 1-channel in, 1-channel sigmoid mask out."""

    def __init__(self, channels=(32, 64, 128, 256)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        in_ch = 1
        for ch in channels:
            self.downs.append(DoubleConv(in_ch, ch))
            in_ch = ch

        self.bottleneck = DoubleConv(channels[-1], channels[-1] * 2)

        rev = list(reversed(channels))
        for ch in rev:
            self.ups.append(nn.ConvTranspose2d(ch * 2, ch, 2, stride=2))
            self.ups.append(DoubleConv(ch * 2, ch))

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


def bce_dice_loss(pred, target, bce_weight: float = 0.5):
    bce = F.binary_cross_entropy(pred, target)
    d = dice_loss(pred, target)
    return bce_weight * bce + (1.0 - bce_weight) * d


@torch.no_grad()
def dice_score(pred, target, threshold: float = 0.5, smooth: float = 1.0):
    pred_bin = (pred > threshold).float()
    inter = (pred_bin * target).sum()
    return float(
        (2.0 * inter + smooth) / (pred_bin.sum() + target.sum() + smooth)
    )


# ---------------------------------------------------------------------
# Baseline training for a single task
# ---------------------------------------------------------------------

def train_baseline(
    task_dict,
    n_shot: int,
    n_epochs: int = 20,   # relatively small, per professor suggestion
    lr: float = 1e-3,
    seed: int = 42,
):
    """
    Train a fresh UNet on n_shot examples of a task.

    Returns (history, model, episode).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    episode = FewShotEpisodeDataset(task_dict, n_shot, seed=seed)
    support_loader = episode.support_loader(batch_size=min(n_shot, 4))
    query_loader = episode.query_loader(batch_size=4)

    model = UNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_dice": []}

    for _ in range(n_epochs):
        model.train()
        running_loss = 0.0
        for imgs, masks in support_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
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
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
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
    n_outer: int = 2000,
    k_inner: int = 5,
    inner_lr: float = 1e-3,
    meta_lr: float = 0.1,
    batch_size: int = 4,
    val_every: int = 200,
    n_shot_val: int = 5,
):
    """
    Reptile meta-training loop with running loss/val logging.

    Returns (meta_model, history) where history has:
      - "outer_step"
      - "val_dice"
      - "meta_loss"
    """
    meta_model = UNet().to(DEVICE)
    task_names = list(train_tasks.keys())
    history = {"outer_step": [], "val_dice": [], "meta_loss": []}

    running_losses = []

    for outer in range(1, n_outer + 1):
        task_name = random.choice(task_names)
        task_dict = train_tasks[task_name]
        dataset = SegDataset(task_dict, augment=True)

        fast = copy.deepcopy(meta_model)
        opt_inner = torch.optim.SGD(fast.parameters(), lr=inner_lr)

        fast.train()
        inner_losses = []
        for _ in range(k_inner):
            idxs = random.sample(
                range(len(dataset)),
                min(batch_size, len(dataset)),
            )
            batch = [dataset[i] for i in idxs]
            imgs = torch.stack([b[0] for b in batch]).to(DEVICE)
            masks = torch.stack([b[1] for b in batch]).to(DEVICE)
            opt_inner.zero_grad()
            preds = fast(imgs)
            loss = bce_dice_loss(preds, masks)
            loss.backward()
            opt_inner.step()
            inner_losses.append(loss.item())

        with torch.no_grad():
            for mp, fp in zip(meta_model.parameters(), fast.parameters()):
                mp.data += meta_lr * (fp.data - mp.data)

        mean_inner = float(np.mean(inner_losses)) if inner_losses else 0.0
        running_losses.append(mean_inner)
        history["meta_loss"].append(mean_inner)

        if outer % 50 == 0:
            print(
                f"[Reptile] outer {outer:5d}/{n_outer} "
                f"inner loss {mean_inner:.4f}"
            )

        if outer % val_every == 0:
            val_d = evaluate_few_shot(
                meta_model,
                val_tasks,
                n_shot=n_shot_val,
                adapt_steps=10,
                adapt_lr=1e-3,
            )
            history["outer_step"].append(outer)
            history["val_dice"].append(val_d)
            print(
                f"[Reptile] outer {outer:5d}/{n_outer} "
                f"val Dice (n_shot={n_shot_val}) {val_d:.4f}"
            )

    return meta_model, history


def adapt_and_evaluate(
    meta_model,
    task_dict,
    n_shot: int,
    adapt_steps: int = 20,
    adapt_lr: float = 1e-3,
    seed: int = 42,
):
    """
    Fine-tune meta_model on n_shot support and evaluate on query set.

    Returns (mean_dice, adapted_model, episode).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    episode = FewShotEpisodeDataset(task_dict, n_shot, seed=seed)
    adapted = copy.deepcopy(meta_model).to(DEVICE)
    opt = torch.optim.Adam(adapted.parameters(), lr=adapt_lr)

    support_loader = episode.support_loader(batch_size=min(n_shot, 4))

    adapted.train()
    for _ in range(adapt_steps):
        for imgs, masks in support_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            opt.zero_grad()
            loss = bce_dice_loss(adapted(imgs), masks)
            loss.backward()
            opt.step()

    adapted.eval()
    dice_vals = []
    with torch.no_grad():
        for imgs, masks in episode.query_loader(batch_size=4):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            dice_vals.append(dice_score(adapted(imgs), masks))

    return float(np.mean(dice_vals)), adapted, episode


def evaluate_few_shot(
    meta_model,
    tasks,
    n_shot: int,
    adapt_steps: int = 10,
    adapt_lr: float = 1e-3,
):
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
    epochs: int = 20,
    lr: float = 1e-3,
    seed: int = 42,
):
    """
    Generic adaptation wrapper used for baselines and meta-inits.

    Uses the same style of inner loop as adapt_and_evaluate but with epochs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    episode = FewShotEpisodeDataset(task_dict, n_shot, seed=seed)
    adapted = copy.deepcopy(base_model).to(DEVICE)
    opt = torch.optim.Adam(adapted.parameters(), lr=lr)
    support_loader = episode.support_loader(batch_size=min(n_shot, 4))

    adapted.train()
    for _ in range(epochs):
        for imgs, masks in support_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            opt.zero_grad()
            loss = bce_dice_loss(adapted(imgs), masks)
            loss.backward()
            opt.step()

    adapted.eval()
    dice_vals = []
    with torch.no_grad():
        for imgs, masks in episode.query_loader(batch_size=4):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            dice_vals.append(dice_score(adapted(imgs), masks))

    return float(np.mean(dice_vals)), adapted, episode