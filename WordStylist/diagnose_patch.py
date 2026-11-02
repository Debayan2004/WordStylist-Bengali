from pathlib import Path
import traceback

target = Path("train.py")
text = target.read_text(encoding="utf-8-sig")
patch = Path("patch_bengali_loader.py").read_text(encoding="utf-8-sig")

try:
    exec(compile(patch, "patch_bengali_loader.py", "exec"), {})
except SyntaxError:
    traceback.print_exc()

print("\nCurrent train.py syntax check:")
compile(text, str(target), "exec")
print("Current train.py is valid.")
