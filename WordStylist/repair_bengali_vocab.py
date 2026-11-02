from pathlib import Path
import pandas as pd
import unicodedata
import json

manifest_path = Path(r"D:\WordStylist\bengali_only_word_manifest_v2.csv")
train_path = Path("train.py")

df = pd.read_csv(manifest_path, encoding="utf-8-sig")
train = df[
    (df["split"] == "train") &
    (df["bengali_only_status_v2"] == "eligible_for_review")
]

characters = sorted({
    char
    for label in train["transcription"].dropna().astype(str)
    for char in unicodedata.normalize("NFC", label)
})

source = train_path.read_text(encoding="utf-8-sig")
start = source.index("MAX_CHARS = 16")
end = source.index("\ndef setup_logging(", start)

replacement = (
    "MAX_CHARS = 16\n"
    "OUTPUT_MAX_LEN = MAX_CHARS\n"
    "c_classes = ''.join(chr(n) for n in "
    + repr([ord(char) for char in characters])
    + ")\n"
    "cdict = {c: i for i, c in enumerate(c_classes)}\n"
    "icdict = {i: c for i, c in enumerate(c_classes)}\n\n"
)

updated = source[:start] + replacement + source[end:]
compile(updated, str(train_path), "exec")
train_path.write_text(updated, encoding="utf-8")

print("Repaired train.py vocabulary source.")
print("Character count:", len(characters))
print("Vocabulary source uses Unicode code points.")
