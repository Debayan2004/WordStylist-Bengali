
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

from GRRNN import GrnnNet


PROJECT_ROOT = Path(r"D:\WordStylist\source")
IMAGE_ROOT = PROJECT_ROOT / "bengali_wordstylist_1000" / "images"
OUTPUT_ROOT = Path(__file__).resolve().parent / "bengali_grrnn_output"
CHECKPOINT = OUTPUT_ROOT / "best_grrnn.pth"
WRITER_MAPPING = PROJECT_ROOT / "writers_dict_train.json"

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 128
NUM_CLASSES = 303
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SyntheticContactSheetDataset(Dataset):
    def __init__(self):
        self.samples = []
        self.words = set()
        self.epochs = set()

        for sheet_path in sorted(IMAGE_ROOT.glob("*.jpg")):
            match = re.fullmatch(r"(.+)_(\d+)", sheet_path.stem)
            if match is None:
                raise ValueError(f"Unexpected filename: {sheet_path.name}")

            word = match.group(1)
            epoch = int(match.group(2))

            with Image.open(sheet_path) as image:
                image = image.convert("L")

                for writer_id in range(16):
                    col = writer_id % 8
                    row = writer_id // 8

                    left = 2 + col * 258
                    top = 2 + row * 66

                    crop = image.crop(
                        (left, top, left + 256, top + 64)
                    )

                    self.samples.append({
                        "image": crop.copy(),
                        "sheet": sheet_path.name,
                        "word": word,
                        "epoch": epoch,
                        "writer_class": writer_id,
                        "writer_id": f"train_{writer_id + 1:03d}"
                    })

            self.words.add(word)
            self.epochs.add(epoch)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = sample["image"]
        original_width, original_height = image.size

        scale = min(
            IMAGE_WIDTH / original_width,
            IMAGE_HEIGHT / original_height
        )

        new_width = max(1, int(round(original_width * scale)))
        new_height = max(1, int(round(original_height * scale)))

        image = image.resize((new_width, new_height),Image.Resampling.LANCZOS)

        image = 255 - np.asarray(image, dtype=np.uint8)

        canvas = np.zeros(
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            dtype=np.float32
        )

        top = (IMAGE_HEIGHT - new_height) // 2
        left = (IMAGE_WIDTH - new_width) // 2

        canvas[
            top:top + new_height,
            left:left + new_width
        ] = image.astype(np.float32) / 255.0

        tensor = torch.from_numpy(canvas).unsqueeze(0)

        return tensor, sample["writer_class"], index


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(IMAGE_ROOT)

    with open(WRITER_MAPPING, "r", encoding="utf-8") as file:
        writer_mapping = json.load(file)

    expected_mapping = {
        f"train_{i:03d}": i - 1 for i in range(1, 17)
    }

    if any(
        writer_mapping.get(writer) != class_id
        for writer, class_id in expected_mapping.items()
    ):
        raise ValueError("Writer mapping does not match expected class IDs.")

    dataset = SyntheticContactSheetDataset()

    if len(dataset) != 1008:
        raise ValueError(
            f"Expected 1008 samples, found {len(dataset)}"
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = GrnnNet(1, NUM_CLASSES, mode="vertical").to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    all_predictions = []
    all_targets = []
    all_top5 = []
    all_confidences = []

    with torch.no_grad():
        for images, labels, indices in loader:
            images = images.to(DEVICE)

            logits = model(images)

            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            probabilities = torch.softmax(logits, dim=1)

            predictions = logits.argmax(dim=1)
            top5 = logits.topk(5, dim=1).indices

            all_predictions.extend(predictions.cpu().tolist())
            all_targets.extend(labels.tolist())
            all_top5.extend(top5.cpu().tolist())
            all_confidences.extend(
                probabilities.max(dim=1).values.cpu().tolist()
            )

    y_true = np.asarray(all_targets)
    y_pred = np.asarray(all_predictions)

    top1 = accuracy_score(y_true, y_pred)

    top5 = np.mean([
        target in predictions
        for target, predictions in zip(y_true, all_top5)
    ])

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(range(16)),
        average="macro",
        zero_division=0
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_ROOT / "synthetic_writer_predictions.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)

        writer.writerow([
            "image_index",
            "contact_sheet",
            "word",
            "epoch",
            "true_writer_class",
            "true_writer_id",
            "predicted_writer_class",
            "predicted_writer_id",
            "confidence",
            "correct",
            "top5_correct"
        ])

        for index, sample in enumerate(dataset.samples):
            prediction = int(y_pred[index])
            target = int(y_true[index])

            writer.writerow([
                index,
                sample["sheet"],
                sample["word"],
                sample["epoch"],
                target,
                sample["writer_id"],
                prediction,
                f"train_{prediction + 1:03d}",
                all_confidences[index],
                int(prediction == target),
                int(target in all_top5[index])
            ])

    per_writer = {}

    for writer_id in range(16):
        mask = y_true == writer_id

        per_writer[f"train_{writer_id + 1:03d}"] = {
            "samples": int(mask.sum()),
            "top1_accuracy": float(
                np.mean(y_pred[mask] == y_true[mask])
            )
        }

    per_word = {}

    for word in sorted(dataset.words):
        indices = [
            i for i, sample in enumerate(dataset.samples)
            if sample["word"] == word
        ]

        word_true = y_true[indices]
        word_pred = y_pred[indices]

        per_word[word] = {
            "samples": len(indices),
            "top1_accuracy": float(
                accuracy_score(word_true, word_pred)
            ),
            "top5_accuracy": float(
                np.mean([
                    word_true[j] in all_top5[indices[j]]
                    for j in range(len(indices))
                ])
            )
        }

    summary = {
        "model": "GR-RNN",
        "checkpoint": str(CHECKPOINT),
        "total_synthetic_images": len(dataset),
        "contact_sheets": len(list(IMAGE_ROOT.glob("*.jpg"))),
        "unique_words": len(dataset.words),
        "writer_classes_evaluated": 16,
        "total_model_classes": NUM_CLASSES,
        "epochs_present": sorted(dataset.epochs),
        "top1_accuracy": float(top1),
        "top5_accuracy": float(top5),
        "macro_f1_16_represented_writers": float(macro_f1),
        "per_writer": per_writer,
        "per_word": per_word
    }

    summary_path = OUTPUT_ROOT / "synthetic_writer_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=4)

    print("\nSYNTHETIC WRITER IDENTIFICATION RESULTS")
    print("=" * 48)
    print(f"Images: {len(dataset)}")
    print(f"Words: {len(dataset.words)}")
    print(f"Writer classes represented: 16 / {NUM_CLASSES}")
    print(f"Top-1 accuracy: {top1:.4%}")
    print(f"Top-5 accuracy: {top5:.4%}")
    print(f"Macro-F1 (16 represented writers): {macro_f1:.4f}")
    print(f"\nPredictions: {csv_path}")
    print(f"Summary: {summary_path}")

    print("\nPER-WRITER ACCURACY")

    for writer_id, result in per_writer.items():
        print(
            f"{writer_id}: "
            f"{result['top1_accuracy']:.2%} "
            f"({result['samples']} samples)"
        )

    print("\nPER-WORD ACCURACY")

    for word, result in per_word.items():
        print(
            f"{word}: "
            f"Top-1={result['top1_accuracy']:.2%}, "
            f"Top-5={result['top5_accuracy']:.2%}"
        )


if __name__ == "__main__":
    main()
