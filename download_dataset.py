from datasets import load_dataset
import pandas as pd
from pathlib import Path

# Download dataset from Hugging Face
ds = load_dataset("Tobi-Bueck/customer-support-tickets")

# Pick a split that exists (often "train")
split_name = "train" if "train" in ds else list(ds.keys())[0]
df = ds[split_name].to_pandas()

# Make a data folder
Path("data").mkdir(exist_ok=True)

# Save raw CSV
out_path = Path("data") / "tickets_raw.csv"
df.to_csv(out_path, index=False)

print(f"✅ Saved {len(df)} rows to {out_path}")
print("Columns:", list(df.columns))
print(df.head(3))
