import os
import pandas as pd

manifest_path = r"D:\WordStylist\bengali_only_word_manifest_v2.csv"
output_path = r"D:\WordStylist\bengali_train_gt.txt"
review_path = r"D:\WordStylist\bengali_train_gt_review.csv"

data = pd.read_csv(manifest_path)
data = data[
    (data["split"] == "train")
    & (data["bengali_only_status_v2"] == "eligible_for_review")
].copy()

data["transcription"] = data["transcription"].fillna("").astype(str).str.strip()
data["image_exists"] = data["image_path"].map(lambda p: os.path.isfile(str(p)))
data["valid_label"] = data["transcription"].map(lambda s: bool(s) and not any(c.isspace() for c in s))
data["valid_writer"] = data["page_id"].map(lambda s: str(s).startswith("train_"))

review = data[~(data["image_exists"] & data["valid_label"] & data["valid_writer"])].copy()
valid = data[data["image_exists"] & data["valid_label"] & data["valid_writer"]].copy()

with open(output_path, "w", encoding="utf-8", newline="\n") as f:
    for _, row in valid.iterrows():
        writer_id = str(row["page_id"])
        image_id = os.path.splitext(str(row["image_name"]))[0]
        transcription = row["transcription"]
        f.write(f"{writer_id},{image_id} {transcription}\n")

review.to_csv(review_path, index=False, encoding="utf-8-sig")

print("Training annotations:", len(valid))
print("Rows requiring review:", len(review))
print("Unique writers:", valid["page_id"].nunique())
print("Output:", output_path)
print("Review file:", review_path)
print("\nFirst five annotations:")
with open(output_path, "r", encoding="utf-8") as f:
    for _, line in zip(range(5), f):
        print(repr(line.rstrip("\n")))
