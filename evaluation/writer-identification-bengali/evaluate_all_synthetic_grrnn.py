
import os
import csv
import json
import numpy as np
import torch

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

from GRRNN import GrnnNet


PROJECT_ROOT = Path(r"D:\WordStylist")

SYNTHETIC_ROOT = (
    PROJECT_ROOT
    / "source"
    / "bengali_wordstylist_1000_full2785"
)

IMAGE_ROOT = SYNTHETIC_ROOT / "images"
METADATA_PATH = SYNTHETIC_ROOT / "metadata.csv"

MAPPING_PATH = PROJECT_ROOT / "source" / "writers_dict_train.json"

OUTPUT_ROOT = Path("bengali_grrnn_full2785_output")
CHECKPOINT = OUTPUT_ROOT / "final_grrnn_full2785.pth"

NUM_CLASSES = 303
EXPECTED_SAMPLES = 2785
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 128
BATCH_SIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SyntheticWriterDataset(Dataset):
    def __init__(self):
        self.samples = []

        with open(METADATA_PATH, "r", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)

            required_columns = {
                "filename",
                "writer_id",
                "class_id",
                "transcription",
                "source_image",
                "checkpoint_epoch"
            }

            if not required_columns.issubset(reader.fieldnames or []):
                raise ValueError("Synthetic metadata is missing required columns")

            for row in reader:
                self.samples.append({
                    "filename": row["filename"],
                    "writer_id": row["writer_id"],
                    "class_id": int(row["class_id"]),
                    "word": row["transcription"],
                    "source_image": row["source_image"],
                    "epoch": int(row["checkpoint_epoch"])
                })

        self.samples.sort(key=lambda x: x["filename"])
        self.words = sorted(set(s["word"] for s in self.samples))

        source_images = [s["source_image"] for s in self.samples]

        if len(source_images) != len(set(source_images)):
            raise ValueError("Duplicate original source images in metadata")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image_path = IMAGE_ROOT / sample["filename"]

        with Image.open(image_path) as image:
            image = image.convert("L")

            original_width, original_height = image.size

            scale = min(
                IMAGE_WIDTH / original_width,
                IMAGE_HEIGHT / original_height
            )

            new_width = max(1, int(round(original_width * scale)))
            new_height = max(1, int(round(original_height * scale)))

            image = image.resize(
                (new_width, new_height),
                Image.Resampling.LANCZOS
            )

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

        return tensor, sample["class_id"], index


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    if not METADATA_PATH.exists():
        raise FileNotFoundError(METADATA_PATH)

    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(IMAGE_ROOT)

    with open(MAPPING_PATH, "r", encoding="utf-8") as file:
        writer_mapping = json.load(file)

    writer_mapping = {
        writer_id: int(class_id)
        for writer_id, class_id in writer_mapping.items()
    }

    if len(writer_mapping) != NUM_CLASSES:
        raise ValueError("Expected exactly 303 writers")

    class_to_writer = {
        class_id: writer_id
        for writer_id, class_id in writer_mapping.items()
    }

    if sorted(class_to_writer) != list(range(NUM_CLASSES)):
        raise ValueError("Invalid writer class mapping")

    dataset = SyntheticWriterDataset()

    if len(dataset) != EXPECTED_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLES} synthetic images, found {len(dataset)}"
        )

    represented_classes = sorted(
        set(sample["class_id"] for sample in dataset.samples)
    )

    if represented_classes != list(range(NUM_CLASSES)):
        raise ValueError("Synthetic dataset does not contain all 303 writers")

    for sample in dataset.samples:
        if writer_mapping.get(sample["writer_id"]) != sample["class_id"]:
            raise ValueError(
                f"Writer mapping mismatch: {sample['writer_id']}"
            )

        if not (IMAGE_ROOT / sample["filename"]).is_file():
            raise FileNotFoundError(IMAGE_ROOT / sample["filename"])

    writer_counts = {
        class_id: sum(
            sample["class_id"] == class_id
            for sample in dataset.samples
        )
        for class_id in range(NUM_CLASSES)
    }

    if sum(writer_counts.values()) != EXPECTED_SAMPLES:
        raise ValueError("Writer sample counts do not sum to 2785")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    model = GrnnNet(
        1,
        NUM_CLASSES,
        mode="vertical"
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )

    if "model_state_dict" not in checkpoint:
        raise ValueError("Invalid GR-RNN checkpoint")

    if checkpoint.get("num_writers") != NUM_CLASSES:
        raise ValueError("Checkpoint writer count does not match")

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    all_predictions = []
    all_targets = []
    all_top5 = []
    all_confidences = []

    with torch.no_grad():
        for images, labels, indices in loader:
            images = images.to(DEVICE, non_blocking=True).float()

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
    top5_predictions = np.asarray(all_top5)

    top1 = accuracy_score(y_true, y_pred)

    top5 = np.mean(
        (top5_predictions == y_true[:, None]).any(axis=1)
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_ROOT / "synthetic_writer_predictions_full2785.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)

        writer.writerow([
            "filename",
            "source_image",
            "word",
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
                sample["filename"],
                sample["source_image"],
                sample["word"],
                target,
                sample["writer_id"],
                prediction,
                class_to_writer[prediction],
                all_confidences[index],
                int(prediction == target),
                int(target in top5_predictions[index])
            ])

    per_writer = {}

    for class_id in range(NUM_CLASSES):
        mask = y_true == class_id
        writer_id = class_to_writer[class_id]

        per_writer[writer_id] = {
            "samples": int(mask.sum()),
            "top1_accuracy": float(
                np.mean(y_pred[mask] == y_true[mask])
            )
        }

    per_word = {}

    for word in dataset.words:
        indices = [
            i for i, sample in enumerate(dataset.samples)
            if sample["word"] == word
        ]

        word_true = y_true[indices]
        word_pred = y_pred[indices]
        word_top5 = top5_predictions[indices]

        per_word[word] = {
            "samples": len(indices),
            "top1_accuracy": float(
                accuracy_score(word_true, word_pred)
            ),
            "top5_accuracy": float(
                np.mean((word_top5 == word_true[:, None]).any(axis=1))
            ),
            "macro_f1": float(
                f1_score(
                    word_true,
                    word_pred,
                    labels=list(range(NUM_CLASSES)),
                    average="macro",
                    zero_division=0
                )
            )
        }

    summary = {
        "model": "GR-RNN",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_epoch": checkpoint["epoch"],
        "training_samples": checkpoint["training_samples"],
        "total_synthetic_images": len(dataset),
        "writer_classes_evaluated": NUM_CLASSES,
        "unique_transcriptions": len(dataset.words),
        "top1_accuracy": float(top1),
        "top5_accuracy": float(top5),
        "macro_f1_303_writers": float(macro_f1),
        "per_writer": per_writer,
        "per_word": per_word
    }

    summary_path = OUTPUT_ROOT / "synthetic_writer_summary_full2785.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=4)

    print("\nFULL SYNTHETIC WRITER IDENTIFICATION RESULTS")
    print("=" * 55)
    print("Images:", len(dataset))
    print("Writer classes:", NUM_CLASSES)
    print("Unique transcriptions:", len(dataset.words))
    print("Samples per writer: minimum", min(writer_counts.values()))
    print("Samples per writer: maximum", max(writer_counts.values()))
    print(f"Top-1 accuracy: {top1:.4%}")
    print(f"Top-5 accuracy: {top5:.4%}")
    print(f"Macro-F1 (303 writers): {macro_f1:.4f}")

    print("\nPER-TRANSCRIPTION RESULTS")

    for word, result in per_word.items():
        print(
            f"{word}: "
            f"Top-1={result['top1_accuracy']:.2%}, "
            f"Top-5={result['top5_accuracy']:.2%}, "
            f"Macro-F1={result['macro_f1']:.4f}, "
            f"Samples={result['samples']}"
        )

    print("\nPredictions:", csv_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
