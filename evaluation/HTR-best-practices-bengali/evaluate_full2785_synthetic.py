
from pathlib import Path
import csv
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import editdistance
from PIL import Image
from omegaconf import OmegaConf

from trainer import HTRTrainer

ROOT = Path(r"D:\WordStylist\source")
DATA_ROOT = ROOT / "bengali_wordstylist_1000_full2785"
IMAGE_ROOT = DATA_ROOT / "images"
METADATA_PATH = DATA_ROOT / "metadata.csv"

CHECKPOINT = Path("saved_models/full2785/final_htrnet_full2785.pt")
OUTPUT_ROOT = Path("synthetic_ocr_evaluation_full2785")

BATCH_SIZE = 32

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

config = OmegaConf.load("config.yaml")
config.train.num_workers = 0
config.eval.num_workers = 0

trainer = HTRTrainer(config)

state_dict = torch.load(
    CHECKPOINT,
    map_location=config.device,
    weights_only=True
)

trainer.net.load_state_dict(state_dict, strict=True)
trainer.net.eval()

metadata = pd.read_csv(METADATA_PATH, encoding="utf-8")

required_columns = {
    "filename",
    "writer_id",
    "transcription"
}

if not required_columns.issubset(metadata.columns):
    raise ValueError(
        f"Missing metadata columns: {required_columns - set(metadata.columns)}"
    )

if metadata["filename"].duplicated().any():
    raise ValueError("Duplicate filenames found in metadata.")

if len(metadata) != 2785:
    raise ValueError(f"Expected 2785 samples, found {len(metadata)}.")

if len(list(IMAGE_ROOT.glob("*.png"))) != 2785:
    raise ValueError("Image count does not match expected count.")

samples = []

for row in metadata.itertuples(index=False):
    image_path = IMAGE_ROOT / row.filename

    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    with Image.open(image_path) as image:
        image = image.convert("L")

        if image.size != (256, 64):
            raise ValueError(
                f"Unexpected dimensions for {row.filename}: {image.size}"
            )

        array = np.asarray(image, dtype=np.float32) / 255.0
        array = 1.0 - array

    samples.append({
        "filename": row.filename,
        "writer_id": row.writer_id,
        "ground_truth": str(row.transcription).strip(),
        "image": array
    })

print("Checkpoint:", CHECKPOINT.resolve())
print("Synthetic samples:", len(samples))
print("Unique writers:", metadata["writer_id"].nunique())
print("Unique transcriptions:", metadata["transcription"].nunique())

results = []
device = torch.device(config.device)
i2c = trainer.classes["i2c"]

with torch.no_grad():
    for start in range(0, len(samples), BATCH_SIZE):
        batch_samples = samples[start:start + BATCH_SIZE]

        batch = np.stack([sample["image"] for sample in batch_samples])
        batch = torch.from_numpy(batch).unsqueeze(1).float().to(device)

        output = trainer.net(batch)

        if isinstance(output, tuple):
            output = output[0]

        predictions = output.argmax(2).permute(1, 0).cpu().tolist()

        for sample, predicted_ids in zip(batch_samples, predictions):
            prediction = trainer.decode(predicted_ids, i2c).strip()
            ground_truth = sample["ground_truth"]

            character_errors = editdistance.eval(
                ground_truth,
                prediction
            )

            results.append({
                "filename": sample["filename"],
                "writer_id": sample["writer_id"],
                "ground_truth": ground_truth,
                "prediction": prediction,
                "character_errors": character_errors,
                "ground_truth_characters": len(ground_truth),
                "cer": character_errors / max(len(ground_truth), 1),
                "word_error": int(ground_truth != prediction)
            })

total_character_errors = sum(
    result["character_errors"] for result in results
)

total_characters = sum(
    result["ground_truth_characters"] for result in results
)

total_word_errors = sum(
    result["word_error"] for result in results
)

total_samples = len(results)

overall_cer = total_character_errors / max(total_characters, 1)
overall_wer = total_word_errors / max(total_samples, 1)

correct_words = total_samples - total_word_errors

csv_path = OUTPUT_ROOT / "synthetic_ocr_predictions_full2785.csv"

with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )
    writer.writeheader()
    writer.writerows(results)

transcription_groups = defaultdict(list)
writer_groups = defaultdict(list)

for result in results:
    transcription_groups[result["ground_truth"]].append(result)
    writer_groups[result["writer_id"]].append(result)

per_transcription = {}

for transcription, group in sorted(transcription_groups.items()):
    errors = sum(item["character_errors"] for item in group)
    characters = sum(item["ground_truth_characters"] for item in group)
    word_errors = sum(item["word_error"] for item in group)

    per_transcription[transcription] = {
        "samples": len(group),
        "cer": errors / max(characters, 1),
        "wer": word_errors / len(group),
        "correct_words": len(group) - word_errors
    }

per_writer = {}

for writer_id, group in sorted(writer_groups.items()):
    errors = sum(item["character_errors"] for item in group)
    characters = sum(item["ground_truth_characters"] for item in group)
    word_errors = sum(item["word_error"] for item in group)

    per_writer[writer_id] = {
        "samples": len(group),
        "cer": errors / max(characters, 1),
        "wer": word_errors / len(group),
        "correct_words": len(group) - word_errors
    }

summary = {
    "checkpoint": str(CHECKPOINT.resolve()),
    "dataset": str(DATA_ROOT.resolve()),
    "samples": total_samples,
    "writers": len(writer_groups),
    "unique_transcriptions": len(transcription_groups),
    "correct_words": correct_words,
    "incorrect_words": total_word_errors,
    "total_character_errors": total_character_errors,
    "total_characters": total_characters,
    "cer": overall_cer,
    "wer": overall_wer,
    "per_transcription": per_transcription,
    "per_writer": per_writer
}

json_path = OUTPUT_ROOT / "synthetic_ocr_summary_full2785.json"

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, ensure_ascii=False, indent=2)

print("\nFULL SYNTHETIC OCR EVALUATION")
print("--------------------------------")
print("Samples:", total_samples)
print("Writers:", len(writer_groups))
print("Unique transcriptions:", len(transcription_groups))
print("Correct words:", correct_words)
print("Incorrect words:", total_word_errors)
print("Character errors:", total_character_errors)
print("Total characters:", total_characters)
print("CER:", f"{overall_cer:.6f}")
print("WER:", f"{overall_wer:.6f}")

print("\nTOP 15 TRANSCRIPTIONS BY SAMPLE COUNT")

for transcription, metrics in sorted(
    per_transcription.items(),
    key=lambda item: item[1]["samples"],
    reverse=True
)[:15]:
    print(
        transcription,
        "| Samples:", metrics["samples"],
        "| CER:", f"{metrics['cer']:.6f}",
        "| WER:", f"{metrics['wer']:.6f}"
    )

print("\nCSV:", csv_path.resolve())
print("Summary:", json_path.resolve())
