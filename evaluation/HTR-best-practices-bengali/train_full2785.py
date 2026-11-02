
import os
import random
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from omegaconf import OmegaConf
from trainer import HTRTrainer
from utils.htr_dataset import HTRDataset
from utils.transforms import aug_transforms

SEED = 42
OUTPUT_DIR = "saved_models/full2785"
CHECKPOINT_NAME = "final_htrnet_full2785.pt"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

config = OmegaConf.load("config.yaml")
config.train.num_workers = 0
config.eval.num_workers = 0

dataset_root = config.data.path
fixed_size = (config.preproc.image_height, config.preproc.image_width)

train_set = HTRDataset(dataset_root, "train", fixed_size=fixed_size, transforms=aug_transforms)
val_set = HTRDataset(dataset_root, "val", fixed_size=fixed_size, transforms=aug_transforms)
test_set = HTRDataset(dataset_root, "test", fixed_size=fixed_size, transforms=aug_transforms)

full_dataset = ConcatDataset([train_set, val_set, test_set])

assert len(train_set) == 2179
assert len(val_set) == 303
assert len(test_set) == 303
assert len(full_dataset) == 2785

print("Training samples:", len(full_dataset))
print("Unique character classes:", len(train_set.character_classes))

loader = DataLoader(
    full_dataset,
    batch_size=config.train.batch_size,
    shuffle=True,
    num_workers=config.train.num_workers,
    drop_last=False
)

config.data.path = dataset_root
trainer = HTRTrainer.__new__(HTRTrainer)
torch.nn.Module.__init__(trainer)
trainer.config = config

classes = np.unique(list(train_set.character_classes) + [" "]).tolist()

trainer.classes = {
    "classes": classes,
    "c2i": {c: i + 1 for i, c in enumerate(classes)},
    "i2c": {i + 1: c for i, c in enumerate(classes)}
}

trainer.prepare_net()
trainer.prepare_losses()
trainer.prepare_optimizers()

trainer.loaders = {"train": loader, "val": loader}
os.makedirs(OUTPUT_DIR, exist_ok=True)

history = []
max_epochs = config.train.num_epochs

print("Starting full-cohort HTRNet training")
print("Epochs:", max_epochs)
print("Batch size:", config.train.batch_size)
print("Device:", config.device)
print("Samples:", len(full_dataset))

for epoch in range(1, max_epochs + 1):
    trainer.train(epoch)
    trainer.scheduler.step()

    if epoch % config.train.save_every_k_epochs == 0 or epoch == max_epochs:
        checkpoint_path = os.path.join(
            OUTPUT_DIR,
            f"htrnet_full2785_epoch_{epoch}.pt"
        )
        torch.save(trainer.net.state_dict(), checkpoint_path)
        print("Checkpoint saved:", checkpoint_path)

    history.append({
        "epoch": epoch,
        "learning_rate": trainer.optimizer.param_groups[0]["lr"]
    })

final_path = os.path.join(OUTPUT_DIR, CHECKPOINT_NAME)
torch.save(trainer.net.state_dict(), final_path)

history_path = os.path.join(OUTPUT_DIR, "training_history.json")
import json
with open(history_path, "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=2)

print("\nFULL-COHORT OCR TRAINING COMPLETED")
print("Training samples:", len(full_dataset))
print("Epochs:", max_epochs)
print("Checkpoint:", final_path)
print("History:", history_path)
