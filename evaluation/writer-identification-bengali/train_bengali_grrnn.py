import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score
from dataloader import DatasetFromFolder
from GRRNN import GrnnNet

PROJECT_ROOT = r"D:\WordStylist"
DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "source",
    "evaluation_models",
    "HTR-best-practices",
    "data",
    "Bengali_WordStylist"
)
MAPPING_PATH = os.path.join(
    PROJECT_ROOT,
    "source",
    "writers_dict_train.json"
)

OUTPUT_DIR = "bengali_grrnn_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
NUM_WORKERS = 0
NUM_WRITERS = 303

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(
            dim=-1,
            index=targets.unsqueeze(1)
        ).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (
            (1.0 - self.smoothing) * nll_loss
            + self.smoothing * smooth_loss
        )
        return loss.mean()

def create_dataset(split):
    return DatasetFromFolder(
        dataset_root=DATASET_ROOT,
        split=split,
        writer_mapping_path=MAPPING_PATH,
        scale_size=(64, 128),
        is_training=(split == "train")
    )

train_set = create_dataset("train")
val_set = create_dataset("val")
test_set = create_dataset("test")

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

model = GrnnNet(
    input_channel=1,
    num_classes=NUM_WRITERS,
    mode="vertical"
).to(device)

criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = StepLR(
    optimizer,
    step_size=10,
    gamma=0.5
)

print("\nGR-RNN CONFIGURATION")
print("--------------------")
print("Device:", device)
print("Writers:", NUM_WRITERS)
print("Training samples:", len(train_set))
print("Validation samples:", len(val_set))
print("Testing samples:", len(test_set))
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Parameters:", sum(p.numel() for p in model.parameters()))

def evaluate(loader):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()

            logits = model(images)
            loss = criterion(logits, labels)

            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            all_targets.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())

    targets = np.asarray(all_targets)
    predictions = np.asarray(all_predictions)

    top1 = np.mean(targets == predictions)

    top5_predictions = np.argsort(
        np.asarray(all_logits), axis=1
    )[:, -5:]

    return total_loss / total_samples, top1, targets, predictions

best_val_accuracy = -1.0
best_epoch = -1
history = []

for epoch in range(1, EPOCHS + 1):
    model.train()

    running_loss = 0.0
    total_samples = 0

    for images, labels, _ in train_loader:
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    train_loss = running_loss / total_samples

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_targets = []
    val_predictions = []
    val_top5_correct = 0

    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()

            logits = model(images)
            loss = criterion(logits, labels)

            predictions = logits.argmax(dim=1)
            top5 = logits.topk(5, dim=1).indices

            val_loss += loss.item() * images.size(0)
            val_correct += (predictions == labels).sum().item()
            val_top5_correct += (
                top5 == labels.unsqueeze(1)
            ).any(dim=1).sum().item()

            val_total += images.size(0)
            val_targets.extend(labels.cpu().tolist())
            val_predictions.extend(predictions.cpu().tolist())

    val_loss /= val_total
    val_top1 = val_correct / val_total
    val_top5 = val_top5_correct / val_total

    val_macro_f1 = f1_score(
        val_targets,
        val_predictions,
        labels=list(range(NUM_WRITERS)),
        average="macro",
        zero_division=0
    )

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch:03d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.5f} | "
        f"Val Loss: {val_loss:.5f} | "
        f"Val Top-1: {val_top1 * 100:.2f}% | "
        f"Val Top-5: {val_top5 * 100:.2f}% | "
        f"Val Macro-F1: {val_macro_f1:.5f} | "
        f"LR: {current_lr:.7f}"
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_top1": val_top1,
        "val_top5": val_top5,
        "val_macro_f1": val_macro_f1,
        "learning_rate": current_lr
    })

    if val_top1 > best_val_accuracy:
        best_val_accuracy = val_top1
        best_epoch = epoch

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_top1": val_top1,
                "val_top5": val_top5,
                "val_macro_f1": val_macro_f1,
                "writer_mapping": train_set.writer_mapping
            },
            os.path.join(OUTPUT_DIR, "best_grrnn.pth")
        )

        print("Best checkpoint saved.")

    scheduler.step()

with open(
    os.path.join(OUTPUT_DIR, "training_history.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(history, f, indent=2)

checkpoint = torch.load(
    os.path.join(OUTPUT_DIR, "best_grrnn.pth"),
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

all_targets = []
all_predictions = []
all_top5 = []

with torch.no_grad():
    for images, labels, _ in test_loader:
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()

        logits = model(images)

        all_targets.extend(labels.cpu().tolist())
        all_predictions.extend(logits.argmax(dim=1).cpu().tolist())
        all_top5.extend(logits.topk(5, dim=1).indices.cpu().tolist())

test_targets = np.asarray(all_targets)
test_predictions = np.asarray(all_predictions)
test_top5 = np.asarray(all_top5)

test_top1 = np.mean(test_targets == test_predictions)
test_top5_accuracy = np.mean(
    (test_top5 == test_targets[:, None]).any(axis=1)
)

test_macro_f1 = f1_score(
    test_targets,
    test_predictions,
    labels=list(range(NUM_WRITERS)),
    average="macro",
    zero_division=0
)

results = {
    "best_epoch": best_epoch,
    "best_validation_top1": float(best_val_accuracy),
    "test_top1": float(test_top1),
    "test_top5": float(test_top5_accuracy),
    "test_macro_f1": float(test_macro_f1),
    "test_samples": len(test_targets),
    "writer_classes": NUM_WRITERS
}

with open(
    os.path.join(OUTPUT_DIR, "test_results.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(results, f, indent=2)

print("\nFINAL TEST RESULTS")
print("------------------")
print("Best epoch:", best_epoch)
print("Test Top-1:", f"{test_top1 * 100:.2f}%")
print("Test Top-5:", f"{test_top5_accuracy * 100:.2f}%")
print("Test Macro-F1:", f"{test_macro_f1:.5f}")
print("Results saved:", os.path.abspath(OUTPUT_DIR))
