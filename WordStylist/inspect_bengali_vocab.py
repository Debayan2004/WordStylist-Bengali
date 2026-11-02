import pandas as pd
import unicodedata
from collections import Counter

path = r"D:\WordStylist\bengali_only_word_manifest_v2.csv"
df = pd.read_csv(path, encoding="utf-8-sig")

train = df[
    (df["split"] == "train") &
    (df["bengali_only_status_v2"] == "eligible_for_review")
].copy()

counter = Counter()
for label in train["transcription"].dropna().astype(str):
    label = unicodedata.normalize("NFC", label)
    counter.update(label)

print("Training samples:", len(train))
print("Unique Unicode characters:", len(counter))
print("\nCharacter inventory:")
for char, count in sorted(counter.items(), key=lambda item: ord(item[0])):
    print(f"U+{ord(char):04X}  {repr(char)}  count={count}")
