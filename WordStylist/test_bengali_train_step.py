
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from torchvision import transforms

from train import Diffusion, EMA, IAMDataset, UNetModel, OUTPUT_MAX_LEN, vocab_size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset_root = r"D:\WordStylist\Bengali_DiffusionPen_Full"
gt_path = r"D:\WordStylist\bengali_train_gt_paths.txt"
vae_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"

with open(gt_path, "r", encoding="utf-8-sig") as f:
    lines = [line.rstrip("\r\n") for line in f if line.strip()]

writer_dict = {}
full_dict = {}

for line in lines:
    first_field, transcription = line.split(" ", 1)
    writer_id, image_path = first_field.split(",", 1)

    if writer_id not in writer_dict:
        writer_dict[writer_id] = len(writer_dict)

    full_dict[len(full_dict)] = {
        "image": image_path,
        "s_id": writer_id,
        "label": transcription
    }

args = type("Args", (), {
    "device": str(device),
    "latent": True,
    "img_feat": True,
    "interpolation": False,
    "channels": 4,
    "emb_dim": 320,
    "num_heads": 4,
    "num_res_blocks": 1,
    "img_size": (64, 256),
    "max_seq_len": OUTPUT_MAX_LEN
})()

image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = IAMDataset(
    full_dict,
    dataset_root,
    writer_dict,
    args,
    transforms=image_transforms
)

loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
images, words, writer_ids = next(iter(loader))

images = images.to(device)
words = words.to(device)
writer_ids = writer_ids.to(device)

vae = AutoencoderKL.from_pretrained(
    vae_path,
    subfolder="vae"
).to(device)
vae.eval()
vae.requires_grad_(False)

model = UNetModel(
    image_size=(64, 256),
    in_channels=4,
    model_channels=320,
    out_channels=4,
    num_res_blocks=1,
    attention_resolutions=(1, 1),
    channel_mult=(1, 1),
    num_heads=4,
    num_classes=len(writer_dict),
    context_dim=320,
    vocab_size=vocab_size,
    args=args,
    max_seq_len=OUTPUT_MAX_LEN
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=0.0001)
loss_fn = nn.MSELoss()
diffusion = Diffusion(img_size=(64, 256), args=args)

with torch.no_grad():
    latents = vae.encode(images.float()).latent_dist.sample() * 0.18215

timesteps = diffusion.sample_timesteps(latents.shape[0]).to(device)
noisy_latents, noise = diffusion.noise_images(latents, timesteps)

model.train()
optimizer.zero_grad(set_to_none=True)

predicted_noise = model(
    noisy_latents,
    original_images=images,
    timesteps=timesteps,
    context=words,
    y=writer_ids,
    or_images=None
)

loss = loss_fn(noise, predicted_noise)

if not torch.isfinite(loss):
    raise RuntimeError("Loss is not finite")

loss.backward()

gradients = [
    parameter.grad
    for parameter in model.parameters()
    if parameter.grad is not None
]

if not gradients:
    raise RuntimeError("No gradients were produced")

if not all(torch.isfinite(gradient).all().item() for gradient in gradients):
    raise RuntimeError("Non-finite gradients detected")

gradient_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)

optimizer.step()

print("Device:", device)
print("Batch images:", tuple(images.shape))
print("Latents:", tuple(latents.shape))
print("Word tokens:", tuple(words.shape))
print("Writer IDs:", writer_ids.tolist())
print("Timesteps:", timesteps.tolist())
print("Initial loss:", loss.item())
print("Gradient norm:", gradient_norm.item())
print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("One-step training test: PASS")
