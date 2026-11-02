
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

root = Path(__file__).resolve().parent / "bengali_grrnn_output"

csv_path = root / "synthetic_writer_predictions.csv"
df = pd.read_csv(csv_path)

results = []

for epoch, group in df.groupby("epoch"):
    y_true = group["true_writer_class"].to_numpy()
    y_pred = group["predicted_writer_class"].to_numpy()

    results.append({
        "epoch": int(epoch),
        "samples": len(group),
        "top1": accuracy_score(y_true, y_pred),
        "top5": group["top5_correct"].mean(),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=list(range(16)),
            average="macro",
            zero_division=0
        )
    })

result_df = pd.DataFrame(results).sort_values("epoch")

output_csv = root / "synthetic_checkpoint_analysis.csv"
result_df.to_csv(output_csv, index=False)

print("\nCHECKPOINT-WISE WRITER IDENTIFICATION")
print(result_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\nBEST CHECKPOINTS")
for metric in ["top1", "top5", "macro_f1"]:
    best = result_df.loc[result_df[metric].idxmax()]
    print(
        f"{metric}: epoch {int(best['epoch'])}, "
        f"{best[metric]:.4f}"
    )

print(f"\nSaved: {output_csv}")

word_results = []

for (epoch, word), group in df.groupby(["epoch", "word"]):
    word_results.append({
        "epoch": int(epoch),
        "word": word,
        "samples": len(group),
        "top1": group["correct"].mean(),
        "top5": group["top5_correct"].mean()
    })

word_df = pd.DataFrame(word_results)
word_path = root / "synthetic_checkpoint_word_analysis.csv"
word_df.to_csv(word_path, index=False)

print(f"Per-word checkpoint analysis: {word_path}")
