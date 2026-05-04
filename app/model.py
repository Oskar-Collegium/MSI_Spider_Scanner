import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

MODEL_PATH = os.getenv("MODEL_PATH", "model/spider_model.pt")

SPECIES = [
    "Black Widow",
    "Blue Tarantula",
    "Bold Jumper",
    "Brown Grass Spider",
    "Brown Recluse Spider",
    "Deinopis Spider",
    "Golden Orb Weaver",
    "Hobo Spider",
    "Huntsman Spider",
    "Ladybird Mimic Spider",
    "Peacock Spider",
    "Red Knee Tarantula",
    "Spiny-backed Orb-weaver",
    "White Kneed Tarantula",
    "Yellow Garden Spider",
]

NUM_CLASSES = len(SPECIES)

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def build_model() -> nn.Module:
    """Build the model architecture (ResNet50 with custom head)."""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def load_model() -> nn.Module | None:
    """Load trained model weights from disk. Returns None if not found."""
    if not os.path.exists(MODEL_PATH):
        return None
    model = build_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def predict(image_bytes: bytes, model: nn.Module) -> dict:
    """
    Run inference on raw image bytes.
    Returns a dict with 'species' and 'confidence'.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return {
        "species": SPECIES[predicted.item()],
        "confidence": round(confidence.item() * 100, 2),
    }
