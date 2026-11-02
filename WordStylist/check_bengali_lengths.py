import pandas as pd
import unicodedata

path = r"D:\WordStylist\bengali_only_word_manifest_v2.csv"
df = pd.read_csv(path, encoding="utf-8-sig")

eligible = df[df["bengali_only_status_v2"] == "eligible_for_review"].copy()
eligible["normalized_label"] = eligible["transcription"].astype(str).map(
    lambda x: unicodedata.normalize("NFC", x)
)
eligible["char_length"] = eligible["normalized_label"].map(len)

print("Eligible samples:", len(eligible))
print("Maximum transcription length:", eligible["char_length"].max())
print("Length distribution:")
print(eligible["char_length"].describe(percentiles=[0.90, 0.95, 0.99]))
print("\nLongest 20 labels:")
print(
    eligible.nlargest(20, "char_length")[
        ["split", "page_id", "image_name", "transcription", "char_length"]
    ].to_string(index=False)
)
