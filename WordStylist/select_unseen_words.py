
import json
import unicodedata
from pathlib import Path

import pandas as pd

root = Path(r"D:\WordStylist")
source = root / "source"

normalize = lambda value: unicodedata.normalize("NFC", str(value).strip())

manifest = pd.read_csv(root / "bengali_only_word_manifest_v2.csv", encoding="utf-8")

test_words = {
    normalize(value)
    for value in manifest.loc[
        manifest["split"].astype(str).str.lower().eq("test"),
        "transcription"
    ].dropna()
}

train_words = set()

with (root / "bengali_train_gt_paths.txt").open("r", encoding="utf-8-sig") as file:
    for line in file:
        parts = line.rstrip("\r\n").split(" ", 1)
        if len(parts) == 2:
            train_words.add(normalize(parts[1]))

with (source / "letter2index.json").open("r", encoding="utf-8") as file:
    vocabulary = set(json.load(file))

candidates = [
    "অজানা", "উদাহরণ", "সহজ", "গগন", "নয়ন",
    "মধুর", "শরণ", "বরণ", "পরশ", "সাগর",
    "কলম", "কমল", "কথা", "কবিতা", "বাংলা",
    "আকাশ", "আলো", "পাখি", "ফসল", "নতুন",
    "গমন", "চরণ", "জলজ", "বসন", "গহন",
    "দরজা", "মেঘ", "বাতাস", "পথিক", "লেখা"
]

selected = []

for candidate in candidates:
    word = normalize(candidate)
    unsupported = set(word) - vocabulary

    if (
        word
        and len(word) <= 10
        and not unsupported
        and word not in train_words
        and word not in test_words
        and word not in selected
    ):
        selected.append(word)

print("Training transcription count:", len(train_words))
print("Test transcription count:", len(test_words))
print("Eligible candidates:", len(selected))

for index, word in enumerate(selected, 1):
    print(f"{index}. {word}")

output = source / "unseen_bengali_candidates.txt"
output.write_text("\n".join(selected[:10]), encoding="utf-8")
print("Saved:", output)
