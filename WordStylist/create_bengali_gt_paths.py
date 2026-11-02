import os
import pandas as pd

manifest_path = r"D:\WordStylist\bengali_only_word_manifest_v2.csv"
output_path = r"D:\WordStylist\bengali_train_gt_paths.txt"
review_path = r"D:\WordStylist\bengali_train_gt_paths_review.csv"
dataset_root = r"D:\WordStylist\Bengali_DiffusionPen_Full"

data = pd.read_csv(manifest_path)
data = data[
    (data["split"] == "train")
    & (data["bengali_only_status_v2"] == "eligible_for_review")
].copy()

data["transcription"] = data["transcription"].fillna("").astype(str).str.strip()
data["relative_path"] = data.apply(
    lambda r: os.path.relpath(str(r["image_path"]), dataset_root).replace("\\", "/"),
    axis=1
)
data["image_exists"] = data["image_path"].map(lambda p: os.path.isfile(str(p)))
data["valid_label"] = data["transcription"].map(
    lambda s: bool(s) and not any(c.isspace() for c in s)
)
data["valid_writer"] = data["page_id"].map(
    lambda s: str(s).startswith("train_")
)
data["valid_path"] = data["relative_path"].map(
    lambda p: p.startswith("train/") and p.endswith(".png")
)

valid_mask = (
    data["image_exists"]
    & data["valid_label"]
    & data["valid_writer"]
    & data["valid_path"]
)

review = data[~valid_mask].copy()
valid = data[valid_mask].copy()

with open(output_path, "w", encoding="utf-8", newline="\n") as f:
    for _, row in valid.iterrows():
        writer_id = str(row["page_id"])
        relative_path = row["relative_path"]
        transcription = row["transcription"]
        f.write(f"{writer_id},{relative_path} {transcription}\n")

review.to_csv(review_path, index=False, encoding="utf-8-sig")

print("Training annotations:", len(valid))
print("Rows requiring review:", len(review))
print("Unique writers:", valid["page_id"].nunique())
print("Unique image paths:", valid["relative_path"].nunique())
print("Output:", output_path)
print("Review file:", review_path)

print("\nFirst five annotations:")
with open(output_path, "r", encoding="utf-8") as f:
    for _, line in zip(range(5), f):
        print(repr(line.rstrip("\n")))
