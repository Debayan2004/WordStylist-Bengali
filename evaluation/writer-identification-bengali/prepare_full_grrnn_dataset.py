import os
import shutil

ROOT = r"D:\WordStylist\source\evaluation_models\HTR-best-practices\data\Bengali_WordStylist"

OUTPUT = os.path.join(ROOT, "full")
OUTPUT_IMAGES = os.path.join(OUTPUT, "images")

os.makedirs(OUTPUT_IMAGES, exist_ok=True)

all_lines = []
seen = set()

for split in ["train", "val", "test"]:
    manifest = os.path.join(ROOT, split, "gt.txt")
    source_images = os.path.join(ROOT, split, "images")

    with open(manifest, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip("\r\n")

        if not line:
            continue

        parts = line.split(maxsplit=1)
        relative_path = parts[0]

        filename = os.path.basename(relative_path)

        if filename not in seen:
            source_path = os.path.join(source_images, filename)

            if not os.path.isfile(source_path):
                if not os.path.splitext(filename)[1]:
                    source_path += ".png"

            if not os.path.isfile(source_path):
                raise FileNotFoundError(source_path)

            destination = os.path.join(OUTPUT_IMAGES, os.path.basename(source_path))
            shutil.copy2(source_path, destination)
            seen.add(filename)

        all_lines.append(line)

if len(all_lines) != 2785:
    raise ValueError(f"Expected 2785 manifest entries, found {len(all_lines)}")

with open(os.path.join(OUTPUT, "gt.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(all_lines) + "\n")

print("Full dataset created")
print("Manifest entries:", len(all_lines))
print("Unique images:", len(seen))
print("Image files:", len(os.listdir(OUTPUT_IMAGES)))
