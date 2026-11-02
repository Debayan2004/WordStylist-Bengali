
import os
from types import SimpleNamespace

import torch
import torchvision
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL

from train import IAMDataset, OUTPUT_MAX_LEN, vocab_size
from unet import UNetModel

root = r"D:\WordStylist\Bengali_DiffusionPen_Full"
gt_path = r"D:\WordStylist\bengali_train_gt_paths.txt"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

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

args = SimpleNamespace(
    device=device,
    interpolation=False,
    img_feat=True,
    channels=4,
    latent=True
)

dataset = IAMDataset(full_dict, root, wr_dict, args, transforms=transforms)
loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

images, words, writer_ids = next(iter(loader))
images = images.to(device)
words = words.to(device)
writer_ids = writer_ids.to(device)

vae = AutoencoderKL.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    subfolder="vae"
).to(device)
vae.eval()
vae.requires_grad_(False)

with torch.no_grad():
    latents = vae.encode(images).latent_dist.sample() * 0.18215

print("Device:", device)
print("Images:", tuple(images.shape))
print("Latents:", tuple(latents.shape))
print("Character tokens:", tuple(words.shape))
print("Writer IDs:", writer_ids.tolist())

model = UNetModel(
    image_size=(64, 256),
    in_channels=4,
    model_channels=320,
    out_channels=4,
    num_res_blocks=1,
    attention_resolutions=(1, 1),
    channel_mult=(1, 1),
    num_heads=4,
    num_classes=len(wr_dict),
    context_dim=320,
    vocab_size=vocab_size,
    args=args,
    max_seq_len=OUTPUT_MAX_LEN
).to(device)

model.eval()
timesteps = torch.randint(0, 1000, (images.shape[0],), device=device)

with torch.no_grad():
    prediction = model(
        latents,
        timesteps=timesteps,
        context=words,
        y=writer_ids
    )

print("UNet output:", tuple(prediction.shape))
print("Output finite:", torch.isfinite(prediction).all().item())
print("UNet smoke test: PASS")
