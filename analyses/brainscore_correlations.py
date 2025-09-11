import torch
import os
import numpy as np
import sklearn
import ndd
import glob
import pandas as pd
import regex as re
from scipy.stats import pearsonr

subset = 10
data_path = f"/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/mi_csv_subset_{subset}.csv"
brainscore_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/leaderboard.csv"
scores_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses"

df_mi = pd.read_csv(data_path)
#print(df_mi)

#extract top MI per model per layer
top_mi = df_mi.loc[df_mi.groupby("model")["mi"].idxmax(), ["model","layer","mi"]].reset_index(drop=True)
#print(top_mi)
#drop models not present in brainscore
top_mi = top_mi.drop(index=[3, 7]).reset_index(drop=True)
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
targets = ["alexnet", "densenet-121", "densenet-169", 
           "densenet-201", "inception_v1", "inception_v3", 
           "resnet_101_v1", "resnet_152_v1", "resnet-18",
           "resnet-34", "resnet_50_v1", "squeezenet1_1",
           "vgg_16"]
mask = df_brainscore[model_col].astype(str).str.strip().str.lower().isin([t.lower() for t in targets])

out = df_brainscore.loc[mask, [model_col] + wanted_cols].copy()

# Nice, consistent column names
rename_map = {inv_norm[k]: k for k in wanted_keys if k in inv_norm}
out = out.rename(columns=rename_map)
#print(out)

#if there are duplicate models, select only the one with the highest "average_vision" score
out_best = out.loc[out.groupby(model_col)["average_vision"].idxmax()].reset_index(drop=True)

# Optional: dictionary form
scores = out_best.set_index(model_col).to_dict(orient="index")
#print(scores)
# Optional: save
out_best.to_csv(os.path.join(scores_path, "best_brainscores.csv"), index=False)

#calculate pearsons r among MI values and the selected brainscores values
combined_df = pd.concat(
    [top_mi.drop(columns=["layer"]).reset_index(drop=True),
    out_best.drop(columns=["Model"]).reset_index(drop=True)],
    axis=1
)
#print(combined)

#columns of brainscore values to correlate against mi 
brainscore_values = [c for c in combined_df.columns if c in {"average_vision", "neural_vision", "behavior_vision"}]
#print(brainscore_values)

rows = []
for metric in brainscore_values:
    #print(metric)
    brain_score = pd.to_numeric(combined_df[metric], errors="coerce")
    #print(brain_score)
    mi_score = pd.to_numeric(combined_df["mi"], errors="coerce")
    #print(mi_score)
    pair = pd.DataFrame({"mi": mi_score, metric: brain_score})
    #print(pair)
    n_values = len(pair)
    #print(n_values)
    if n_values >= 2:
        r, p = pearsonr(pair["mi"].astype(float), pair[metric].astype(float))
    rows.append({"metric": metric, "pearson_r": r, "n": n_values, "p_value": p})

corr_df = pd.DataFrame(rows)#.sort_values("metric").reset_index(drop=True)
corr_df.to_csv(os.path.join(scores_path, f"mi_brainscores_corr_{subset}.csv"), index=False)
print(corr_df)