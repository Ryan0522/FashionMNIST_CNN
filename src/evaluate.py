from __future__ import annotations
import os
import math
import itertools
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def confusion_matrix(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int = 10) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()
            true = y.numpy()
            for t, p in zip(true, pred):
                cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.maximum(cm_sum, 1)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        text = f"{cm[i, j]}\n({cm_norm[i, j]:.2f})"
        ax.text(j, i, text, ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def save_misclassified_grid(model: nn.Module, loader: DataLoader, device: torch.device, out_path: str, n: int = 25) -> None:
    ensure_dir(os.path.dirname(out_path))
    model.eval()
    images, labels, preds = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            p = logits.argmax(dim=1).cpu()
            mism = p.ne(y)
            idxs = torch.where(mism)[0]
            for k in idxs:
                images.append(x[k].cpu())
                labels.append(int(y[k].item()))
                preds.append(int(p[k].item()))
                if len(images) >= n:
                    break
            if len(images) >= n:
                break

    if len(images) == 0:
        return
    grid_size = int(math.ceil(math.sqrt(len(images))))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
    axes = np.array(axes).reshape(grid_size, grid_size)
    for ax in axes.flat:
        ax.axis('off')

    for idx, (img, t, p) in enumerate(zip(images, labels, preds)):
        r, c = divmod(idx, grid_size)
        ax = axes[r, c]
        ax.imshow(img.squeeze(0), cmap='gray')
        ax.set_title(f"T:{t} / P:{p}", fontsize=8)
        ax.axis('off')

    ensure_dir(os.path.dirname(out_path))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def visualize_conv_filters(model: nn.Module, out_dir: str) -> None:
    """Save conv weight grids for up to 3 conv layers."""
    ensure_dir(out_dir)

    def _save_kernels(weight: torch.Tensor, save_path: str) -> None:
        w = weight.detach().cpu().clone()
        if w.size(1) == 1:
            w = w[:, 0]
        w_min = w.amin(dim=(-1, -2), keepdim=True)
        w_max = w.amax(dim=(-1, -2), keepdim=True)
        w = (w - w_min) / torch.clamp(w_max - w_min, min=1e-6)
        n = int(math.ceil(math.sqrt(w.size(0))))
        fig, axes = plt.subplots(n, n, figsize=(n * 1.8, n * 1.8))
        axes = np.array(axes).reshape(n, n)
        for ax in axes.flat:
            ax.axis('off')
        for i in range(w.size(0)):
            r, c = divmod(i, n)
            axes[r, c].imshow(w[i].cpu().numpy(), cmap='gray')
            axes[r, c].axis('off')
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    conv_layers = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            conv_layers.append((name, m))

    conv_layers = conv_layers[:3]
    for i, (name, m) in enumerate(conv_layers, start=1):
        save_path = os.path.join(out_dir, f"filters_conv{i}.png")
        _save_kernels(m.weight, save_path)