from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class LogisticRegressionNet(nn.Module):
    def __init__(self, in_features: int = 28 * 28, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        return self.fc(x)
    
class MLP(nn.Module):
    def __init__(self, in_features: int = 28 * 28, hidden1: int = 256, hidden2: int = 128, num_classes: int = 10, p: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, hidden1),
        nn.ReLU(inplace=True),
        nn.Dropout(p),
        nn.Linear(hidden1, hidden2),
        nn.ReLU(inplace=True),
        nn.Dropout(p),
        nn.Linear(hidden2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class CNNBaseline(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2) # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(2) # 28x28 -> 14x14
        self.fc = nn.Linear(16 * 14 * 14, num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
class CNNAdvanced(nn.Module):
    def __init__(self, num_classes: int = 10, p: float = 0.3):
        super().__init__()
        self.block1 = nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2), # 28 -> 14
        nn.Dropout(p),
        )
        self.block2 = nn.Sequential(
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2), # 14 -> 7
        nn.Dropout(p),
        )
        self.head = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p),
        nn.Linear(128, num_classes),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.head(x)
        return x
    
def build_model(name: str, num_classes: int = 10) -> nn.Module:
    name = name.lower()
    if name in {"logreg", "logistic", "logistic_regression"}:
        return LogisticRegressionNet(num_classes=num_classes)
    if name == "mlp":
        return MLP(num_classes=num_classes)
    if name in {"cnn", "baseline_cnn"}:
        return CNNBaseline(num_classes=num_classes)
    if name in {"adv", "advanced", "cnn_adv", "cnnadvanced"}:
        return CNNAdvanced(num_classes=num_classes)
    raise ValueError(f"Unknown model name: {name}")