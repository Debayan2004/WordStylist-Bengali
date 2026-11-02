from pathlib import Path
import shutil

target = Path("train.py")
backup = Path("train.py.bengali_backup")
text = target.read_text(encoding="utf-8-sig")

old_path = "img_path = os.path.join(self.image_path, image_name)"
new_path = 'img_path = os.path.join(self.image_path, *image_name.replace("\\\\", "/").split("/"))'

start_marker = "        with open(args.gt_train, 'r') as f:"
end_marker = "        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)"

if text.count(old_path) != 1:
    raise RuntimeError("Image-path line not found exactly once.")

if text.count(start_marker) != 1 or text.count(end_marker) != 1:
    raise RuntimeError("Annotation block markers not found exactly once.")

start = text.index(start_marker)
end = text.index(end_marker, start) + len(end_marker)
old_block = text[start:end]

if "full_dict[idx]" not in old_block or "number of train writer styles" not in old_block:
    raise RuntimeError("Unexpected original annotation block.")

new_block = '''        with open(args.gt_train, 'r', encoding='utf-8-sig') as f:
            train_data = [line.rstrip('\\r\\n') for line in f if line.strip()]

        wr_dict = {}
        full_dict = {}
        wr_index = 0

        for line in train_data:
            first_field, transcription = line.split(' ', 1)
            s_id, image = first_field.split(',', 1)

            if s_id not in wr_dict:
                wr_dict[s_id] = wr_index
                wr_index += 1

            full_dict[len(full_dict)] = {
                'image': image,
                's_id': s_id,
                'label': transcription
            }

        print('number of training samples', len(full_dict))
        print('number of train writer styles', len(wr_dict))
        style_classes = len(wr_dict)

        with open("writers_dict_train.json", "w", encoding="utf-8") as f:
            json.dump(wr_dict, f, ensure_ascii=False)

        train_ds = IAMDataset(
            full_dict,
            args.iam_path,
            wr_dict,
            args,
            transforms=transforms
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )'''

updated = text.replace(old_path, new_path, 1)
updated = updated[:start] + new_block + updated[end:]
compile(updated, str(target), "exec")

if not backup.exists():
    shutil.copy2(target, backup)

target.write_text(updated, encoding="utf-8")
print("Patch applied successfully.")
