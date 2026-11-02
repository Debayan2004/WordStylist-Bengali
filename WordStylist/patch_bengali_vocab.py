from pathlib import Path
import json
import unicodedata
import pandas as pd

target = Path("train.py")
backup = Path("train.py.bengali_vocab_backup")

text = target.read_text(encoding="utf-8-sig")

if "c_classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'" not in text:
    raise RuntimeError("The original English vocabulary was not found. No changes made.")

if "def labelDictionary():" not in text or "def label_padding(labels, num_tokens):" not in text:
    raise RuntimeError("The expected vocabulary functions were not found. No changes made.")

manifest_path = Path(r"D:\WordStylist\bengali_only_word_manifest_v2.csv")
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

if not characters:
    raise RuntimeError("No training characters were found. No changes made.")

new_vocabulary = '''MAX_CHARS = 16
OUTPUT_MAX_LEN = MAX_CHARS
c_classes = {chars}
cdict = {{c: i for i, c in enumerate(c_classes)}}
icdict = {{i: c for i, c in enumerate(c_classes)}}

def setup_logging(args):
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'images'), exist_ok=True)

def label_padding(labels, num_tokens):
    labels = unicodedata.normalize('NFC', labels)
    if len(labels) > OUTPUT_MAX_LEN:
        raise ValueError(f"Label length {{len(labels)}} exceeds OUTPUT_MAX_LEN={{OUTPUT_MAX_LEN}}: {{labels!r}}")
    unknown = sorted(set(labels) - set(letter2index))
    if unknown:
        raise ValueError(f"Unknown characters in label {{labels!r}}: {{unknown!r}}")
    ids = [letter2index[char] for char in labels]
    ids.extend([tokens["PAD_TOKEN"]] * (OUTPUT_MAX_LEN - len(ids)))
    return ids

def labelDictionary():
    letter2index = {{label: n for n, label in enumerate(c_classes)}}
    index2letter = {{v: k for k, v in letter2index.items()}}
    with open("letter2index.json", "w", encoding="utf-8") as f:
        json.dump(letter2index, f, ensure_ascii=False, indent=2)
    with open("index2letter.json", "w", encoding="utf-8") as f:
        json.dump(index2letter, f, ensure_ascii=False, indent=2)
    return len(c_classes), letter2index, index2letter

char_classes, letter2index, index2letter = labelDictionary()
tokens = {{"PAD_TOKEN": char_classes}}
num_tokens = 1
vocab_size = char_classes + 1

print("num_tokens", num_tokens)
print("num of character classes", char_classes)
print("padding token ID", tokens["PAD_TOKEN"])
print("vocabulary size", vocab_size)
'''.format(chars=repr(''.join(characters)))

start = text.index("MAX_CHARS = 10")
end = text.index("\ndef save_images(", start)

updated = text[:start] + new_vocabulary + text[end:]
updated = updated.replace(
    "import json\n",
    "import json\nimport unicodedata\n",
    1
)

compile(updated, str(target), "exec")

if not backup.exists():
    backup.write_text(text, encoding="utf-8")

target.write_text(updated, encoding="utf-8")
print("Vocabulary patch applied.")
print("Training characters:", len(characters))
print("Maximum sequence length:", 16)
print("Padding ID:", len(characters))
