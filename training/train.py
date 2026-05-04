"""
Training script for the Spider Scanner model.
Run from the project root:
    python training/train.py --data_dir data/spiders --epochs 20

Dataset directory structure expected:
    data/spiders/
        train/
            Black Widow/
            Blue Tarantula/
            ... (one folder per species)
        valid/
            Black Widow/
            ...
        test/
            Black Widow/
            ...
        spiders.csv
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Freeze all layers except the final FC head
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate(model, loader, device) -> float:
    """Return accuracy (%) on a dataloader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            _, preds = torch.max(model(images), 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total * 100


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tf, eval_tf = get_transforms()

    train_dir = os.path.join(args.data_dir, "train")
    valid_dir = os.path.join(args.data_dir, "valid")
    test_dir  = os.path.join(args.data_dir, "test")

    train_set = datasets.ImageFolder(train_dir, transform=train_tf)
    valid_set = datasets.ImageFolder(valid_dir, transform=eval_tf)

    num_classes = len(train_set.classes)
    print(f"Classes found ({num_classes}): {train_set.classes}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total * 100
        train_loss = running_loss / total
        val_acc = evaluate(model, valid_loader, device)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.1f}% | "
              f"Val Acc: {val_acc:.1f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            torch.save(model.state_dict(), args.output)
            print(f"  ✓ New best model saved ({val_acc:.1f}%)")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to: {args.output}")

    # --- Final test set evaluation ---
    if os.path.isdir(test_dir):
        print("\nEvaluating on test set...")
        test_set    = datasets.ImageFolder(test_dir, transform=eval_tf)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

        # Load the best saved weights for test evaluation
        model.load_state_dict(torch.load(args.output, map_location=device))
        test_acc = evaluate(model, test_loader, device)
        print(f"Test Accuracy: {test_acc:.1f}%")
        if test_acc >= 85.0:
            print("✓ Meets the 85% accuracy requirement.")
        else:
            print("✗ Below 85% target — consider more epochs or unfreezing deeper layers.")
    else:
        print(f"No test directory found at '{test_dir}', skipping test evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Spider Scanner model")
    parser.add_argument("--data_dir",   type=str, default="data/spiders",      help="Path to dataset root")
    parser.add_argument("--output",     type=str, default="model/spider_model.pt", help="Output path for model weights")
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
