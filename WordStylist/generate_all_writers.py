
import os
import csv
import json
import random
import unicodedata
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL
from tqdm import tqdm

from train import Diffusion, label_padding, num_tokens, vocab_size, OUTPUT_MAX_LEN
from unet import UNetModel


ROOT = r"D:\WordStylist\source"
CHECKPOINT = os.path.join(ROOT, "bengali_wordstylist_1000", "models", "ema_ckpt.pt")
WRITER_MAPPING = os.path.join(ROOT, "writers_dict_train.json")
MANIFEST = r"D:\WordStylist\bengali_train_gt_paths.txt"
ORIGINAL_ROOT = r"D:\WordStylist\Bengali_DiffusionPen_Full"
VAE_PATH = r"C:\Users\cvpru\.cache\huggingface\hub\models--runwayml--stable-diffusion-v1-5\snapshots\451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
OUTPUT_DIR = os.path.join(ROOT, "bengali_wordstylist_1000_full2785")

BATCH_SIZE = 16
EXPECTED_SAMPLES = 2785
DEVICE = "cuda:0"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if os.path.exists(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
    raise RuntimeError(
        f"Output directory is not empty: {OUTPUT_DIR}. "
        "Use a fresh output directory to avoid mixing generations."
    )

IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

with open(WRITER_MAPPING, "r", encoding="utf-8") as f:
    writer_mapping = json.load(f)

if len(writer_mapping) != 303:
    raise ValueError(f"Expected 303 writers, found {len(writer_mapping)}")

writer_to_class = {k: int(v) for k, v in writer_mapping.items()}

if sorted(writer_to_class.values()) != list(range(303)):
    raise ValueError("Writer class IDs are not exactly 0 to 302")

samples_by_text = defaultdict(list)
seen_paths = set()

with open(MANIFEST, "r", encoding="utf-8-sig") as f:
    for line_number, line in enumerate(f, start=1):
        line = line.rstrip("\r\n")

        if not line:
            continue

        metadata, separator, transcription = line.partition(" ")

        if not separator or not transcription:
            raise ValueError(f"Invalid manifest line: {line_number}")

        writer_id, separator, relative_path = metadata.partition(",")

        if not separator or not relative_path:
            raise ValueError(f"Invalid metadata on line: {line_number}")

        if writer_id not in writer_to_class:
            raise ValueError(f"Unknown writer {writer_id} on line {line_number}")

        if relative_path in seen_paths:
            raise ValueError(f"Duplicate source path: {relative_path}")

        seen_paths.add(relative_path)

        source_path = os.path.join(ORIGINAL_ROOT, relative_path)

        if not os.path.isfile(source_path):
            raise FileNotFoundError(source_path)

        normalized_text = unicodedata.normalize("NFC", transcription)

        if len(normalized_text) > OUTPUT_MAX_LEN:
            raise ValueError(
                f"Transcription exceeds model limit on line {line_number}: "
                f"{transcription}"
            )

        label_padding(normalized_text, num_tokens)

        samples_by_text[normalized_text].append({
            "writer_id": writer_id,
            "class_id": writer_to_class[writer_id],
            "source_image": relative_path,
            "transcription": transcription
        })

total_samples = sum(len(items) for items in samples_by_text.values())

if total_samples != EXPECTED_SAMPLES:
    raise ValueError(
        f"Expected {EXPECTED_SAMPLES} samples, found {total_samples}"
    )

if len(seen_paths) != EXPECTED_SAMPLES:
    raise ValueError("Source image count does not match expected sample count")

args = SimpleNamespace(
    device=DEVICE,
    img_size=(64, 256),
    channels=4,
    emb_dim=320,
    num_res_blocks=1,
    num_heads=4,
    latent=True,
    img_feat=True,
    interpolation=False,
    stable_dif_path=VAE_PATH
)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

unet = UNetModel(
    image_size=args.img_size,
    in_channels=args.channels,
    model_channels=args.emb_dim,
    out_channels=args.channels,
    num_res_blocks=args.num_res_blocks,
    attention_resolutions=(1, 1),
    channel_mult=(1, 1),
    num_heads=args.num_heads,
    num_classes=303,
    context_dim=args.emb_dim,
    vocab_size=vocab_size,
    args=args,
    max_seq_len=OUTPUT_MAX_LEN
).to(DEVICE)

state = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=True
)

unet.load_state_dict(state, strict=True)
unet.eval()
unet.requires_grad_(False)

vae = AutoencoderKL.from_pretrained(
    VAE_PATH,
    subfolder="vae",
    local_files_only=True
).to(DEVICE)

vae.eval()
vae.requires_grad_(False)

diffusion = Diffusion(
    noise_steps=1000,
    beta_start=1e-4,
    beta_end=0.02,
    img_size=args.img_size,
    args=args
)

print("EMA checkpoint loaded successfully")
print("UNet parameters:", sum(p.numel() for p in unet.parameters()))
print("Writers:", len(writer_mapping))
print("Original samples:", total_samples)
print("Unique transcriptions:", len(samples_by_text))
print("Batch size:", BATCH_SIZE)

metadata_path = os.path.join(OUTPUT_DIR, "metadata.csv")
sample_number = 0

with open(metadata_path, "w", newline="", encoding="utf-8-sig") as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow([
        "filename",
        "writer_id",
        "class_id",
        "transcription",
        "source_image",
        "checkpoint_epoch"
    ])

    for word, word_samples in tqdm(
        samples_by_text.items(),
        total=len(samples_by_text),
        desc="Transcriptions"
    ):
        for start in range(0, len(word_samples), BATCH_SIZE):
            batch_samples = word_samples[start:start + BATCH_SIZE]

            labels = torch.tensor(
                [sample["class_id"] for sample in batch_samples],
                dtype=torch.long,
                device=DEVICE
            )

            count = len(batch_samples)

            unet.eval()

            with torch.inference_mode():
                generated = diffusion.sampling(
                    model=unet,
                    vae=vae,
                    n=count,
                    x_text=word,
                    labels=labels,
                    args=args,
                    cfg_scale=3
                )

            generated = generated.detach().cpu().float()

            if generated.ndim != 4 or generated.shape[1] != 3:
                raise ValueError(f"Unexpected output shape: {generated.shape}")

            if generated.shape[2:] != (64, 256):
                raise ValueError(f"Unexpected image dimensions: {generated.shape}")

            for index, sample in enumerate(batch_samples):
                sample_number += 1

                filename = f"sample_{sample_number:05d}.png"
                image_path = os.path.join(IMAGE_DIR, filename)

                array = generated[index].permute(1, 2, 0).numpy()
                array = np.clip(array * 255.0, 0, 255).astype(np.uint8)

                Image.fromarray(array).save(image_path)

                writer.writerow([
                    filename,
                    sample["writer_id"],
                    sample["class_id"],
                    sample["transcription"],
                    sample["source_image"],
                    1000
                ])

            csvfile.flush()

            del generated

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

print("\nGeneration completed")
print("Generated images:", sample_number)
print("Expected images:", EXPECTED_SAMPLES)
print("Metadata:", metadata_path)
print("Image directory:", IMAGE_DIR)
