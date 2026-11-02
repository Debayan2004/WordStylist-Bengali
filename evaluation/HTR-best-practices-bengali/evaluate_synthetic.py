from pathlib import Path
import re
import csv
import numpy as np
import torch
import editdistance
from PIL import Image
from collections import Counter
from omegaconf import OmegaConf
from trainer import HTRTrainer

config = OmegaConf.load("config.yaml")
config.train.num_workers = 0
config.eval.num_workers = 0

trainer = HTRTrainer(config)

checkpoint = Path("saved_models/best_htrnet.pt")
state_dict = torch.load(checkpoint, map_location=config.device, weights_only=True)
trainer.net.load_state_dict(state_dict, strict=True)
trainer.net.eval()

generated_root = Path(r"D:\WordStylist\source\bengali_wordstylist_1000\images")
output_root = Path("synthetic_ocr_evaluation")
crop_root = output_root / "crops"
crop_root.mkdir(parents=True, exist_ok=True)

i2c = trainer.classes["i2c"]
samples = []

for path in sorted(generated_root.glob("*.jpg")):
    match = re.match(r"^(.+)_([0-9]+)$", path.stem)

    if match is None:
        continue

    word = match.group(1)

    with Image.open(path) as image:
        image = image.convert("L")

        if image.size != (2066, 134):
            raise ValueError(f"Unexpected dimensions: {path.name}: {image.size}")

        for row in range(2):
            for col in range(8):
                x0 = 2 + col * 258
                y0 = 2 + row * 66

                crop = image.crop((x0, y0, x0 + 256, y0 + 64))

                crop_name = f"{path.stem}_panel_{row * 8 + col + 1:02d}.png"
                crop.save(crop_root / crop_name)

                array = np.asarray(crop, dtype=np.float32) / 255.0
                array = 1.0 - array

                samples.append({
                    "source_sheet": path.name,
                    "filename": crop_name,
                    "ground_truth": word,
                    "image": array
                })

print("Contact sheets:", len(list(generated_root.glob("*.jpg"))))
print("Individual samples:", len(samples))
print("Samples per word:", Counter(s["ground_truth"] for s in samples))
print("Input dimensions:", samples[0]["image"].shape)

results = []
device = torch.device(config.device)
batch_size = 16

with torch.no_grad():
    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start:start + batch_size]

        batch = np.stack([s["image"] for s in batch_samples])
        batch = torch.from_numpy(batch).unsqueeze(1).float().to(device)

        output = trainer.net(batch)

        if isinstance(output, tuple):
            output = output[0]

        predictions = output.argmax(2).permute(1, 0).cpu().tolist()

        for sample, predicted_ids in zip(batch_samples, predictions):
            prediction = trainer.decode(predicted_ids, i2c).strip()
            ground_truth = sample["ground_truth"].strip()

            errors = editdistance.eval(ground_truth, prediction)

            results.append({
                "source_sheet": sample["source_sheet"],
                "filename": sample["filename"],
                "ground_truth": ground_truth,
                "prediction": prediction,
                "character_errors": errors,
                "ground_truth_characters": len(ground_truth),
                "cer": errors / max(len(ground_truth), 1),
                "word_error": int(ground_truth != prediction)
            })

total_errors = sum(r["character_errors"] for r in results)
total_chars = sum(r["ground_truth_characters"] for r in results)
total_word_errors = sum(r["word_error"] for r in results)

cer = total_errors / max(total_chars, 1)
wer = total_word_errors / max(len(results), 1)

csv_path = output_root / "synthetic_ocr_predictions.csv"

with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

summary_path = output_root / "synthetic_ocr_summary.txt"

with open(summary_path, "w", encoding="utf-8") as f:
    f.write(f"Checkpoint: {checkpoint}\n")
    f.write(f"Contact sheets: {len(list(generated_root.glob('*.jpg')))}\n")
    f.write(f"Individual samples: {len(results)}\n")
    f.write(f"Unique words: {len(set(r['ground_truth'] for r in results))}\n")
    f.write(f"Total character errors: {total_errors}\n")
    f.write(f"Total characters: {total_chars}\n")
    f.write(f"CER: {cer:.6f}\n")
    f.write(f"WER: {wer:.6f}\n")

print("\nSYNTHETIC OCR RESULTS")
print("---------------------")
print("Individual samples:", len(results))
print("Correct words:", len(results) - total_word_errors)
print("Incorrect words:", total_word_errors)
print("CER:", f"{cer:.6f}")
print("WER:", f"{wer:.6f}")

print("\nPER-WORD RESULTS")

for word in sorted(set(r["ground_truth"] for r in results)):
    subset = [r for r in results if r["ground_truth"] == word]
    word_errors = sum(r["character_errors"] for r in subset)
    word_chars = sum(r["ground_truth_characters"] for r in subset)
    word_wrong = sum(r["word_error"] for r in subset)

    print(
        repr(word),
        "| Samples:", len(subset),
        "| CER:", f"{word_errors / max(word_chars, 1):.6f}",
        "| WER:", f"{word_wrong / len(subset):.6f}"
    )

print("\nCSV:", csv_path.resolve())
print("Summary:", summary_path.resolve())
print("Cropped images:", crop_root.resolve())
