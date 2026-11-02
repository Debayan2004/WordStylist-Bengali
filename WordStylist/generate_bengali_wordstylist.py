import os
import json
import random
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from diffusers import AutoencoderKL

from train import Diffusion, vocab_size, OUTPUT_MAX_LEN
from unet import UNetModel

ROOT = Path(r"D:\WordStylist")
SOURCE = ROOT / "source"
MODEL_DIR = SOURCE / "bengali_wordstylist_1000" / "models"
OUTPUT_DIR = SOURCE / "bengali_wordstylist_1000" / "inference"
IMAGE_DIR = OUTPUT_DIR / "generated_words"
CANDIDATE_FILE = SOURCE / "unseen_bengali_candidates.txt"
WRITER_FILE = SOURCE / "writers_dict_train.json"
VOCAB_FILE = SOURCE / "letter2index.json"
TEST_MANIFEST = ROOT / "bengali_only_word_manifest_v2.csv"
TRAIN_GT = ROOT / "bengali_train_gt_paths.txt"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

normalize = lambda value: unicodedata.normalize("NFC", str(value).strip())

with VOCAB_FILE.open("r", encoding="utf-8") as file:
    vocabulary = set(json.load(file))

with WRITER_FILE.open("r", encoding="utf-8") as file:
    writer_dict = json.load(file)

if len(writer_dict) != 303:
    raise RuntimeError(f"Expected 303 writer styles, found {len(writer_dict)}")

with TRAIN_GT.open("r", encoding="utf-8-sig") as file:
    train_words = set()
    for line in file:
        parts = line.rstrip("\r\n").split(" ", 1)
        if len(parts) == 2:
            train_words.add(normalize(parts[1]))

import pandas as pd

manifest = pd.read_csv(TEST_MANIFEST, encoding="utf-8")
test_words = {
    normalize(value)
    for value in manifest.loc[
        manifest["split"].astype(str).str.lower().eq("test"),
        "transcription"
    ].dropna()
}

candidates = [
    normalize(word)
    for word in CANDIDATE_FILE.read_text(encoding="utf-8").splitlines()
    if normalize(word)
]

words = []
for word in candidates:
    if (
        word not in words
        and word not in train_words
        and word not in test_words
        and len(word) <= OUTPUT_MAX_LEN
        and set(word).issubset(vocabulary)
    ):
        words.append(word)
    if len(words) == 10:
        break

if len(words) < 10:
    raise RuntimeError(
        f"Only {len(words)} verified candidates were found. "
        "Add more independent Bengali candidates before generating."
    )

snapshot_root = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--stable-diffusion-v1-5--stable-diffusion-v1-5"
    / "snapshots"
)

snapshots = sorted(
    [path for path in snapshot_root.iterdir() if path.is_dir() and path.name != ".no_exist"],
    key=lambda path: path.stat().st_mtime,
    reverse=True
)

if not snapshots:
    raise FileNotFoundError(f"No cached Stable Diffusion snapshot found in {snapshot_root}")

vae_path = snapshots[0]

args = SimpleNamespace(
    device=device,
    img_size=(64, 256),
    channels=4,
    emb_dim=320,
    num_heads=4,
    num_res_blocks=1,
    latent=True,
    img_feat=True,
    interpolation=False,
    wandb_log=False,
    stable_dif_path=str(vae_path),
    save_path=str(OUTPUT_DIR)
)

model = UNetModel(
    image_size=args.img_size,
    in_channels=args.channels,
    model_channels=args.emb_dim,
    out_channels=args.channels,
    num_res_blocks=args.num_res_blocks,
    attention_resolutions=(1, 1),
    channel_mult=(1, 1),
    num_heads=args.num_heads,
    num_classes=len(writer_dict),
    context_dim=args.emb_dim,
    vocab_size=vocab_size,
    args=args,
    max_seq_len=OUTPUT_MAX_LEN
).to(device)

checkpoint_path = MODEL_DIR / "ema_ckpt.pt"
if not checkpoint_path.exists():
    raise FileNotFoundError(f"EMA checkpoint not found: {checkpoint_path}")

state = torch.load(checkpoint_path, map_location=device, weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()

vae = AutoencoderKL.from_pretrained(
    str(vae_path),
    subfolder="vae",
    local_files_only=True
).to(device)
vae.requires_grad_(False)
vae.eval()

diffusion = Diffusion(img_size=args.img_size, args=args)

writer_id = 0
labels = torch.tensor([writer_id], dtype=torch.long, device=device)

print("Device:", device)
print("EMA checkpoint:", checkpoint_path)
print("VAE snapshot:", vae_path)
print("Writer style ID:", writer_id)
print("Selected words:", words)

generated = []

for index, word in enumerate(words, 1):
    print(f"Generating {index}/10: {word}")
    torch.manual_seed(1000 + index)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1000 + index)

    with torch.inference_mode():
        result = diffusion.sampling(
            model,
            vae,
            n=1,
            x_text=word,
            labels=labels,
            args=args,
            mix_rate=1,
            cfg_scale=0
        )

    array = result[0].detach().float().cpu().permute(1, 2, 0).numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, mode="RGB")
    image_path = IMAGE_DIR / f"{index:02d}_generated.png"
    image.save(image_path)
    generated.append((word, image.copy(), image_path))
    print("Saved:", image_path)

font_candidates = [
    Path(r"C:\Windows\Fonts\Nirmala.ttf"),
    Path(r"C:\Windows\Fonts\NirmalaB.ttf"),
    Path(r"C:\Windows\Fonts\Vrinda.ttf")
]

font = ImageFont.load_default()
for font_path in font_candidates:
    if font_path.exists():
        try:
            font = ImageFont.truetype(str(font_path), 26)
            break
        except OSError:
            pass

row_height = 112
sheet_width = 1100
sheet = Image.new("RGB", (sheet_width, 90 + row_height * len(generated)), "white")
draw = ImageDraw.Draw(sheet)
draw.text((24, 20), "WordStylist Bengali Word Generation", fill="black", font=font)
draw.text((24, 55), "Target text and generated handwriting", fill="black", font=font)

for index, (word, image, image_path) in enumerate(generated):
    y = 90 + index * row_height
    draw.line((20, y, sheet_width - 20, y), fill=(215, 215, 215), width=1)
    draw.text((28, y + 35), word, fill="black", font=font)
    draw.text((250, y + 8), "Generated", fill=(80, 80, 80), font=font)

    preview = image.copy()
    preview.thumbnail((780, 82))
    sheet.paste(preview, (250, y + 30))

draw.line((20, sheet.height - 1, sheet_width - 20, sheet.height - 1), fill=(215, 215, 215))
sheet_path = OUTPUT_DIR / "bengali_wordstylist_unseen_comparison.png"
pdf_path = OUTPUT_DIR / "bengali_wordstylist_unseen_comparison.pdf"
sheet.save(sheet_path)
sheet.convert("RGB").save(pdf_path, "PDF", resolution=150)

(OUTPUT_DIR / "selected_words.txt").write_text(
    "\n".join(words) + "\n",
    encoding="utf-8"
)

print("Comparison PNG:", sheet_path)
print("Comparison PDF:", pdf_path)
print("Individual images:", IMAGE_DIR)
print("Finished.")
