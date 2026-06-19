# 2-3 Minute Demo Narration

## 0:00-0:20 - Problem

"This project is about few-shot abdominal MRI segmentation. The goal is to segment a new anatomical structure when we only have 1, 3, or 5 labelled examples, because medical annotations are expensive."

## 0:20-0:45 - Data

"Each anatomical structure is treated as its own task. We meta-train on the training tasks, validate on separate tasks, and only evaluate finally on held-out test tasks. Here the grayscale image is the MRI slice, and the colored overlay is the segmentation mask."

## 0:45-1:10 - Model

"The backbone is a compact U-Net. It takes a one-channel MRI slice and outputs a one-channel sigmoid mask. Reptile and the baseline use the same architecture, so the comparison is about the initialization rather than model capacity."

## 1:10-1:45 - Reptile

"During Reptile meta-training, we sample a training task, clone the U-Net, run a short inner loop on that task, then move the meta-weights toward the task-adapted weights. Repeating this teaches the initial weights to be close to many task-specific solutions."

## 1:45-2:10 - Few-Shot Adaptation

"At test time, a held-out task provides only a few support slices. The meta-initialized model adapts for 30 steps, and the remaining query slices are used for Dice evaluation."

## 2:10-2:40 - Results

"Qualitatively, we can inspect the MRI, ground truth, Reptile prediction, and baseline prediction side by side. Quantitatively, Reptile outperforms the from-scratch baseline on both held-out structures across 1-shot, 3-shot, and 5-shot settings."

## 2:40-3:00 - Conference Extension Preview and Close

"As a brief preview of where the work went next, this 2D few-shot project was extended with a group for a conference-paper direction on the larger dataset. The extension moves from PNG slices to full NIfTI volumes and a 3D CNN/U-Net style architecture, scaling the segmentation pipeline toward richer 3D scan context."

## Recording Path

1. Open `video_demo/demo.html` in a browser.
2. Record full screen or browser window.
3. Scroll through one section per narration block.
4. Optional: run `python video_demo/export_tensorboard.py`, then `tensorboard --logdir video_demo/tensorboard_logs` if you prefer recording TensorBoard's Images and Scalars tabs.
