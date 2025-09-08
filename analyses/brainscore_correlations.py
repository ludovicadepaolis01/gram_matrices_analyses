import torch
import os
import numpy as np
import sklearn
import ndd
import glob
import pandas as pd
import regex as re

data_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/mi_csv.csv"
brainscore_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/leaderboard.csv"

df_mi = pd.read_csv(data_path)
#print(df_mi)

#extract top MI per model per layer
top_mi = df_mi.loc[df_mi.groupby("model")["mi"].idxmax(), ["model","layer","mi"]].reset_index(drop=True)
#print(top_mi)

#open brainscore leaderboard which is a bit problematic
df_brainscore = pd.read_csv(
                brainscore_path,
                comment="#",          # drops the "Generated on ..." line
                sep=None,             # auto-detects comma vs tab
                engine="python",
                encoding="utf-8-sig",
                na_values=["", "NaN", "nan", "—", "–"]  # treat dashes as NaN
)

# Clean column names
df_brainscore.columns = df_brainscore.columns.str.strip()

# 3) Make everything numeric except 'Model' (and any other obvious text cols)
for col in df_brainscore.columns:
    if col.lower() not in {"model"}:
        # remove any stray non-numeric chars (NBSPs, etc.) before conversion
        df_brainscore[col] = pd.to_numeric(
            df_brainscore[col].astype(str).str.replace(r"[^\d\.\-eE]", "", regex=True),
            errors="coerce"
        )

model_col = next(c for c in df_brainscore.columns if c.strip().lower() in {"model", "model_name", "name"})

# Normalize column names to match keys like "average_vision"
norm = {c: c.strip().lower().replace(" ", "_") for c in df_brainscore.columns}
inv_norm = {v: k for k, v in norm.items()}

wanted_keys = ["average_vision", "neural_vision", "behavior_vision"]
wanted_cols = [inv_norm[k] for k in wanted_keys if k in inv_norm]

# Filter the two models (case-insensitive, exact match after strip)
targets = ["vgg_16", "alexnet"]
mask = df_brainscore[model_col].astype(str).str.strip().str.lower().isin([t.lower() for t in targets])

out = df_brainscore.loc[mask, [model_col] + wanted_cols].copy()

# Nice, consistent column names
rename_map = {inv_norm[k]: k for k in wanted_keys if k in inv_norm}
out = out.rename(columns=rename_map)

print(out)
# Optional: dictionary form
scores = out.set_index(model_col).to_dict(orient="index")
print(scores)
# Optional: save
out.to_csv("vgg16_alexnet_brainscores.csv", index=False)

