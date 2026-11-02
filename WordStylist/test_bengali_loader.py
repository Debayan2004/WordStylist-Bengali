
import os
import torch
import torchvision
from train import IAMDataset

root = r"D:\WordStylist\Bengali_DiffusionPen_Full"
gt_path = r"D:\WordStylist\bengali_train_gt_paths.txt"

with open(gt_path, "r", encoding="utf-8-sig") as f:
    lines = [line.rstrip("\r\n") for line in f if line.strip()]

wr_dict = {}
full_dict = {}

for line in lines:
    first_field, transcription = line.split(" ", 1)
    writer_id, image_path = first_field.split(",", 1)

    if writer_id not in wr_dict:
        wr_dict[writer_id] = len(wr_dict)

    full_dict[len(full_dict)] = {
        "image": image_path,
        "s_id": writer_id,
        "label": transcription
    }

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = IAMDataset(full_dict, root, wr_dict, None, transforms=transforms)

print("Samples:", len(dataset))
print("Writers:", len(wr_dict))

for i in range(min(3, len(dataset))):
    image, word, writer_id = dataset[i]
    print(
        f"Sample {i}: image_shape={tuple(image.shape)}, "
        f"word_shape={tuple(word.shape)}, "
        f"writer_id={writer_id.item()}, "
        f"image_finite={torch.isfinite(image).all().item()}"
    )
