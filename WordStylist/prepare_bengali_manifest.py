
import os
import re
import unicodedata
import pandas as pd
from PIL import Image

DATASET_ROOT = r"D:\WordStylist\Bengali_DiffusionPen_Full"
EXCEL_PATH = r"D:\WordStylist\diffusionpen_bengali_annotation_with_ocr_final_annotation.xlsx"
OUTPUT_DIR = r"D:\WordStylist"

df = pd.read_excel(EXCEL_PATH, sheet_name="Annotations")
df["transcription"] = df["transcription"].fillna("").astype(str).map(
    lambda x: unicodedata.normalize("NFC", x).strip()
)

results = []

for _, row in df.iterrows():
    split = str(row["split"]).strip().lower()
    page_id = str(row["page_id"]).strip()
    image_name = str(row["image_name"]).strip()

    if not image_name.lower().endswith(".png"):
        image_name += ".png"

    image_path = os.path.join(DATASET_ROOT, split, page_id, "all_words", image_name)
    label = row["transcription"]
    reasons = []

    if not os.path.isfile(image_path):
        reasons.append("missing_image")
        width = height = None
    else:
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                if image.mode not in ("L", "RGB", "RGBA"):
                    reasons.append("unusual_image_mode")
        except Exception:
            width = height = None
            reasons.append("unreadable_image")

    if not label:
        reasons.append("empty_label")

    if any(char.isspace() for char in label):
        reasons.append("contains_whitespace")

    if len(label) > 20:
        reasons.append("very_long_label")
    elif len(label) > 10:
        reasons.append("long_label")

    if re.search(r"[^\u0980-\u09FF\s.,!?;:'\"()\-০-৯0-9]", label):
        reasons.append("non_bengali_or_unexpected_character")

    if str(row.get("confirmed", "")).strip().lower() != "confirmed":
        reasons.append("not_confirmed")

    results.append({
        "split": split,
        "page_id": page_id,
        "image_name": image_name,
        "image_path": image_path,
        "transcription": label,
        "label_length": len(label),
        "width": width,
        "height": height,
        "status": "review" if reasons else "candidate",
        "review_reasons": ";".join(reasons)
    })

manifest = pd.DataFrame(results)
manifest = manifest.drop_duplicates(
    subset=["split", "page_id", "image_name"],
    keep="first"
)

manifest_path = os.path.join(OUTPUT_DIR, "bengali_word_manifest.csv")
manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

print("Manifest saved:", manifest_path)
print("\nSamples by split and status:")
print(manifest.groupby(["split", "status"]).size().to_string())
print("\nReview reasons:")
print(
    manifest.loc[manifest["status"] == "review", "review_reasons"]
    .str.split(";")
    .explode()
    .value_counts()
    .to_string()
)
print("\nImage dimensions:")
print(manifest.groupby(["split", "width", "height"]).size().to_string())
