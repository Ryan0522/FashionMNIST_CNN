from __future__ import annotations
import os
import argparse
from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from models import build_model
from evaluate import (
    ensure_dir,
    count_parameters,
    evaluate,
    confusion_matrix,
    plot_confusion_matrix,
    save_misclassified_grid,
    visualize_conv_filters,
)


CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def get_loaders(data_dir: str, batch_size: int, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_ds = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=tfm)
    test_ds = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def plot_curves(train_losses, val_losses, train_accs, val_accs, out_dir: str, tag: str) -> None:
    ensure_dir(out_dir)
    # Loss
    plt.figure()
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"loss_curve_{tag}.png"), dpi=150)
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(train_accs, label='train')
    plt.plot(val_accs, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"accuracy_curve_{tag}.png"), dpi=150)
    plt.close()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Fashion-MNIST training")
    parser.add_argument('--model', type=str, default='cnn', choices=['logreg', 'mlp', 'cnn', 'adv'], help='which model to train')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--results_dir', type=str, default='results/final')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--step_size', type=int, default=0, help='StepLR step_size (0 to disable)')
    parser.add_argument('--gamma', type=float, default=0.1, help='StepLR gamma')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_misclassified', action='store_true')
    parser.add_argument('--mis_n', type=int, default=25)
    parser.add_argument('--save_filters', action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device)
    ensure_dir(args.results_dir)

    # Data
    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size, args.num_workers)

    # Model
    model = build_model(args.model)
    model.to(device)

    tag = args.model.lower()
    print(f"Model: {args.model} | Params: {count_parameters(model):,}")

    # Optimizer / Loss / Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.step_size and args.step_size > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Train
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, test_loader, device)
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)

        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")

    # Curves
    plot_curves(train_losses, val_losses, train_accs, val_accs, args.results_dir, tag)

    # Confusion Matrix
    cm = confusion_matrix(model, test_loader, device, num_classes=len(CLASS_NAMES))
    plot_confusion_matrix(cm, CLASS_NAMES, os.path.join(args.results_dir, f"confusion_matrix_{tag}.png"))

    # Misclassified grid (optional)
    if args.save_misclassified:
        save_misclassified_grid(model, test_loader, device, os.path.join(args.results_dir, 'misclassified', f'misclassified_{tag}.png'), n=args.mis_n)

    # Filters (optional, only meaningful for conv nets)
    if args.save_filters:
        visualize_conv_filters(model, args.results_dir)


if __name__ == "__main__":
    main()
