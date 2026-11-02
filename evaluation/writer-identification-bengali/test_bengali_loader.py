import os
import torch
from torch.utils.data import DataLoader
from dataloader import DatasetFromFolder

project_root = r"D:\WordStylist"
dataset_root = os.path.join(
    project_root,
    "source",
    "evaluation_models",
    "HTR-best-practices",
    "data",
    "Bengali_WordStylist"
)

mapping_path = os.path.join(
    project_root,
    "source",
    "writers_dict_train.json"
)

for split in ["train", "val", "test"]:
    dataset = DatasetFromFolder(
        dataset_root=dataset_root,
        split=split,
        writer_mapping_path=mapping_path,
        scale_size=(64, 128),
        is_training=(split == "train")
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    images, labels, filenames = next(iter(loader))

    print("\n", split.upper())
    print("Image shape:", images.shape)
    print("Label shape:", labels.shape)
    print("Image range:", images.min().item(), images.max().item())
    print("Writer labels:", labels.tolist())
    print("Filenames:", list(filenames))

    assert images.shape == (4, 1, 64, 128)
    assert labels.min().item() >= 0
    assert labels.max().item() < 303

print("\nDATA LOADER TEST PASSED")
