
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

from sklearn.metrics import accuracy_score

from dataloader import DatasetFromFolder
from GRRNN import GrnnNet


DATASET_ROOT = r"D:\WordStylist\source\evaluation_models\HTR-best-practices\data\Bengali_WordStylist"
MAPPING_PATH = r"D:\WordStylist\source\writers_dict_train.json"
OUTPUT_DIR = "bengali_grrnn_full2785_output"

SEED = 42
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
NUM_WORKERS = 0
NUM_WRITERS = 303
EXPECTED_SAMPLES = 2785

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

dataset = DatasetFromFolder(
    dataset_root=DATASET_ROOT,
    split="full",
    writer_mapping_path=MAPPING_PATH,
    scale_size=(64, 128),
    is_training=True
)

if len(dataset) != EXPECTED_SAMPLES:
    raise ValueError(
        f"Expected {EXPECTED_SAMPLES} samples, found {len(dataset)}"
    )

if len(set(sample[1] for sample in dataset.samples)) != NUM_WRITERS:
    raise ValueError("Dataset does not contain all 303 writer classes")

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=False
)

model = GrnnNet(
    input_channel=1,
    num_classes=NUM_WRITERS,
    mode="vertical"
).to(device)

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
        return (
            (1.0 - self.smoothing) * nll_loss
            + self.smoothing * smooth_loss
        ).mean()


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

print("\nFULL-COHORT GR-RNN TRAINING")
print("---------------------------")
print("Device:", device)
print("Training samples:", len(dataset))
print("Writer classes:", NUM_WRITERS)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Parameters:", sum(p.numel() for p in model.parameters()))

history = []

for epoch in range(1, EPOCHS + 1):
    model.train()

    running_loss = 0.0
    total_samples = 0
    correct = 0

    for images, labels, _ in train_loader:
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        correct += (predictions == labels).sum().item()

    train_loss = running_loss / total_samples
    train_accuracy = correct / total_samples
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch:03d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.5f} | "
        f"Train Accuracy: {train_accuracy * 100:.2f}% | "
        f"LR: {current_lr:.7f}"
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "learning_rate": current_lr
    })

    scheduler.step()

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "writer_mapping": dataset.writer_mapping,
        "num_writers": NUM_WRITERS,
        "training_samples": EXPECTED_SAMPLES
    }

    torch.save(
        checkpoint,
        os.path.join(OUTPUT_DIR, "latest_grrnn_full2785.pth")
    )

    if epoch == EPOCHS:
        torch.save(
            checkpoint,
            os.path.join(OUTPUT_DIR, "final_grrnn_full2785.pth")
        )

with open(
    os.path.join(OUTPUT_DIR, "training_history.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(history, f, indent=2)

print("\nFULL-COHORT TRAINING COMPLETED")
print("Final epoch:", EPOCHS)
print("Final training loss:", history[-1]["train_loss"])
print("Final training accuracy:", history[-1]["train_accuracy"])
print("Checkpoint:", os.path.abspath(
    os.path.join(OUTPUT_DIR, "final_grrnn_full2785.pth")
))
print("History:", os.path.abspath(
    os.path.join(OUTPUT_DIR, "training_history.json")
))
