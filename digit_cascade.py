"""
digit_cascade.py

Train 10 binary classifiers (one per digit) on MNIST, save them, and run cascade inference:
model_0 -> model_1 -> ... -> model_9

Usage examples:
  # Train all models (takes time)
  python digit_cascade.py --train-all --epochs 5 --batch-size 256

  # Train a single digit (e.g., digit 2)
  python digit_cascade.py --train-digit 2 --epochs 5

  # Test cascade on MNIST test set
  python digit_cascade.py --test-cascade --models-dir models

  # Predict a single image file (scanned digit or handwritten photo)
  python digit_cascade.py --predict-file some_digit.jpg --models-dir models
"""

import os
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

# -------------------------------------------------------
# Model
# -------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> 7x7
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)  # single logit for binary classification
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # returns logits shape (N,)

# -------------------------------------------------------
# Binary MNIST wrapper (label 1 if target == digit else 0)
# -------------------------------------------------------
class BinaryMNIST(Dataset):
    def __init__(self, root, digit, train=True, transform=None, download=True):
        self.mnist = MNIST(root=root, train=train, download=download)
        self.digit = int(digit)
        self.transform = transform

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        bin_label = 1 if label == self.digit else 0
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(bin_label, dtype=torch.float32)

# -------------------------------------------------------
# Training function for one digit
# -------------------------------------------------------
def train_one_digit(digit, epochs=5, batch_size=128, lr=1e-3, models_dir="models", device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"Training digit {digit} on device {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = BinaryMNIST(root="./data", digit=digit, train=True, transform=transform, download=True)
    val_size = 5000
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in tqdm(train_loader, desc=f"Digit {digit} Epoch {epoch}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * imgs.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()

        avg_loss = total_loss / total
        acc = correct / total
        print(f"Epoch {epoch} Train Loss: {avg_loss:.4f}, Train Acc: {acc:.4f}")

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss += float(loss.item()) * imgs.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.numel()
        print(f"Epoch {epoch} Val Loss: {val_loss/val_total:.4f}, Val Acc: {val_correct/val_total:.4f}")

    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, f"digit_{digit}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved model for digit {digit} -> {save_path}")
    return save_path

# -------------------------------------------------------
# Train all digits sequentially
# -------------------------------------------------------
def train_all_digits(epochs=3, batch_size=128, lr=1e-3, models_dir="models"):
    for d in range(10):
        train_one_digit(digit=d, epochs=epochs, batch_size=batch_size, lr=lr, models_dir=models_dir)

# -------------------------------------------------------
# Load all models
# -------------------------------------------------------
def load_all_models(models_dir="models", device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    models = []
    for d in range(10):
        model = SimpleCNN().to(device)
        path = os.path.join(models_dir, f"digit_{d}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}. Train it first.")
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)
    return models, device

# -------------------------------------------------------
# Preprocess external image
# -------------------------------------------------------
def prepare_image_for_mnist(path_or_pil, device=None):
    if isinstance(path_or_pil, str):
        img = Image.open(path_or_pil).convert("L")
    else:
        img = path_or_pil.convert("L")

    # Pillow >=10: use Resampling.LANCZOS
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.ANTIALIAS

    img = img.resize((28, 28), resample)

    # auto-invert
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.mean() > 0.5:
        img = ImageOps.invert(img)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tensor = transform(img).unsqueeze(0)
    return tensor

# -------------------------------------------------------
# Cascade prediction logic
# -------------------------------------------------------
def cascade_predict_image(img_path, models_dir="models", threshold=0.5):
    models, device = load_all_models(models_dir=models_dir)
    x = prepare_image_for_mnist(img_path).to(device)

    with torch.no_grad():
        for digit, model in enumerate(models):
            logit = model(x).item()
            prob = 1.0 / (1.0 + np.exp(-logit))
            print(f"Model {digit} prob={prob:.4f}")
            if prob >= threshold:
                return digit, prob

    probs = []
    with torch.no_grad():
        for model in models:
            logit = model(x).item()
            probs.append(1.0 / (1.0 + np.exp(-logit)))
    best = int(np.argmax(probs))
    return best, float(probs[best])

# -------------------------------------------------------
# Evaluate cascade
# -------------------------------------------------------
def evaluate_cascade_on_mnist_test(models_dir="models", threshold=0.5, limit=None):
    models, device = load_all_models(models_dir=models_dir)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    testset = MNIST(root="./data", train=False, download=True, transform=transform)
    loader = DataLoader(testset, batch_size=1, shuffle=False)
    correct = 0
    total = 0
    for i, (img, label) in enumerate(loader):
        if limit and i >= limit:
            break
        img = img.to(device)
        chosen = None
        with torch.no_grad():
            for digit, model in enumerate(models):
                prob = torch.sigmoid(model(img)).item()
                if prob >= threshold:
                    chosen = digit
                    break
        if chosen is None:
            with torch.no_grad():
                probs = [torch.sigmoid(m(img)).item() for m in models]
            chosen = int(np.argmax(probs))
        if chosen == int(label.item()):
            correct += 1
        total += 1
    print(f"Cascade accuracy on test set (first {total} samples): {correct}/{total} = {correct/total:.4f}")
    return correct/total

# -------------------------------------------------------
# CLI
# -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-digit", type=int, help="Train model for one digit 0..9")
    parser.add_argument("--train-all", action="store_true", help="Train all digit models")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--test-cascade", action="store_true", help="Evaluate cascade on MNIST test set")
    parser.add_argument("--predict-file", type=str, help="Predict a single image file using cascade")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for 'yes' in each binary model")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test samples when evaluating")

    args = parser.parse_args()

    if args.train_all:
        train_all_digits(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, models_dir=args.models_dir)

    if args.train_digit is not None:
        train_one_digit(digit=args.train_digit, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, models_dir=args.models_dir)

    if args.test_cascade:
        evaluate_cascade_on_mnist_test(models_dir=args.models_dir, threshold=args.threshold, limit=args.limit)

    if args.predict_file:
        digit, prob = cascade_predict_image(args.predict_file, models_dir=args.models_dir, threshold=args.threshold)
        print(f"Cascade predicted digit: {digit} (prob {prob:.4f})")
