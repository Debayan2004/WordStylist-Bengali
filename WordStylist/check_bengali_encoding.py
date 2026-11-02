import json
from pathlib import Path

root = Path(r"D:\WordStylist")
gt_path = root / "bengali_train_gt_paths.txt"
vocab_path = root / "source" / "letter2index.json"

with gt_path.open("r", encoding="utf-8") as f:
    lines = f.readlines()

with vocab_path.open("r", encoding="utf-8") as f:
    vocab = json.load(f)

print("GT FILE:", gt_path)
print("TOTAL LINES:", len(lines))
print("FIRST GT LINE:")
print(repr(lines[0]))
print("FIRST TRANSCRIPTION:")
parts = lines[0].rstrip("\r\n").split(",", 1)
if len(parts) == 2:
    fields = parts[1].split(" ", 1)
    if len(fields) == 2:
        text = fields[1]
        print(text)
        print("UNICODE:", [hex(ord(c)) for c in text])

print("\nVOCABULARY SIZE:", len(vocab))
print("FIRST 15 VOCABULARY CHARACTERS:")
for char, idx in list(vocab.items())[:15]:
    print(idx, repr(char), hex(ord(char[0])))

print("\nIMAGE CHECK:")
if len(parts) == 2:
    relative_path = parts[1].split(" ", 1)[0]
    image_path = root / "Bengali_DiffusionPen_Full" / relative_path
    print("IMAGE:", image_path)
    print("EXISTS:", image_path.exists())

print("\nWRITER COUNT:", len(json.loads(
    (root / "source" / "writers_dict_train.json").read_text(encoding="utf-8")
)))
