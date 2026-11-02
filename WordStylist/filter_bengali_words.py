
import os
import unicodedata
import pandas as pd
from PIL import Image

INPUT_PATH = r"D:\WordStylist\bengali_word_manifest.csv"
OUTPUT_PATH = r"D:\WordStylist\bengali_word_filtered_manifest.csv"
MAX_LABEL_LENGTH = 20

df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig", keep_default_na=False)
results = []

allowed_punctuation = set(".,!?;:'\"()[]{}-–—/")

for _, row in df.iterrows():
    label = unicodedata.normalize("NFC", str(row["transcription"])).strip()
    image_path = str(row["image_path"])
    reasons = []

    if not label:
        reasons.append("empty_label")

    if any(char.isspace() for char in label):
        reasons.append("contains_whitespace")

    if len(label) > MAX_LABEL_LENGTH:
        reasons.append("label_too_long")

    for char in label:
        category = unicodedata.category(char)
        code = ord(char)

        is_bengali = 0x0980 <= code <= 0x09FF
        is_bengali_danda = code in (0x0964, 0x0965)
        is_ascii_digit = "0" <= char <= "9"
        is_bengali_digit = 0x09E6 <= code <= 0x09EF
        is_punctuation = char in allowed_punctuation
        is_combining_mark = category.startswith("M")

        if not (
            is_bengali
            or is_bengali_danda
            or is_ascii_digit
            or is_bengali_digit
            or is_punctuation
            or is_combining_mark
        ):
            reasons.append("unexpected_character")
            break

    try:
        if not os.path.isfile(image_path):
            reasons.append("missing_image")
        else:
            with Image.open(image_path) as image:
                image.verify()
    except Exception:
        reasons.append("unreadable_image")

    results.append({
        **row.to_dict(),
        "transcription": label,
        "word_filter_status": "eligible_for_review" if not reasons else "excluded",
        "word_filter_reasons": ";".join(reasons)
    })

filtered = pd.DataFrame(results)
filtered.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("Saved:", OUTPUT_PATH)
print("\nCounts by split and status:")
print(filtered.groupby(["split", "word_filter_status"]).size().to_string())
print("\nExclusion reasons:")
excluded = filtered[filtered["word_filter_status"] == "excluded"]
print(
    excluded["word_filter_reasons"]
    .str.split(";")
    .explode()
    .value_counts()
    .to_string()
)
