import torch
import os
import numpy as np
import sklearn
import ndd
import glob
import pandas as pd
import regex as re
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

subset = "all"
data_path = f"/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/mi_csv_subset_{subset}_k47.csv"
brainscore_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/leaderboard.csv"
scores_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses"
plot_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/plots"

df_mi = pd.read_csv(data_path)

#extract top MI per model per layer
top_mi = df_mi.loc[df_mi.groupby("model")["mi"].idxmax(), ["model","layer","mi"]].reset_index(drop=True)
#print(top_mi)
#drop models not present in brainscore
top_mi = top_mi.reset_index(drop=True)
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
#print(norm)
inv_norm = {v: k for k, v in norm.items()}
#print(inv_norm)

wanted_keys = ["average_vision", "neural_vision", "behavior_vision", "v1", "v2", "v4", "it"]
#print(wanted_keys)
wanted_cols = [inv_norm[k] for k in wanted_keys if k in inv_norm]
#print(wanted_cols)

# Filter the two models (case-insensitive, exact match after strip)
targets = ["alexnet", "densenet-121", "densenet-169", 
           "densenet-201", "inception_v1", "inception_v3", 
           "resnet_101_v1", "resnet_152_v1", "resnet-18",
           "resnet-34", "resnet_50_v1", "mobilenet_v2_1-4_224_pytorch",
           "vgg_16", "vgg_19"]
mask = df_brainscore[model_col].astype(str).str.strip().str.lower().isin([t.lower() for t in targets])

out = df_brainscore.loc[mask, [model_col] + wanted_cols].copy()
#print(out)

# Nice, consistent column names
rename_map = {inv_norm[k]: k for k in wanted_keys if k in inv_norm}
out = out.rename(columns=rename_map)
#print(out)

#if there are duplicate models, select only the one with the highest "average_vision" score
out_best = out.loc[out.groupby(model_col)["average_vision"].idxmax()].reset_index(drop=True)
#print(out_best)
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
#print(combined_df)

#columns of brainscore values to correlate against mi 
brainscore_values = [c for c in combined_df.columns if c in {"average_vision", "neural_vision", "behavior_vision", "v1", "v2", "v4", "it"}]
#print(brainscore_values)

mi_all = pd.to_numeric(combined_df["mi"], errors="coerce")
model_all = combined_df["model"].astype(str)
valid = mi_all.notna()
tick_pos = mi_all[valid].values
tick_labels = [f"{x:.3g}\n{m}" for x, m in zip(tick_pos, model_all[valid])]
#colors = plt.cm.tab10.colors  # distinct colors

rows = []
for index, metric in enumerate(brainscore_values):

    #print(metric)
    brain_score = pd.to_numeric(combined_df[metric], errors="coerce")
    #print(brain_score)
    mi_score = pd.to_numeric(combined_df["mi"], errors="coerce")
    #print(mi_score)
    pair = pd.DataFrame({"model": combined_df["model"],
                        "mi": mi_score, 
                        metric: brain_score})
    n_values = len(pair)
    #print(n_values)
    if n_values >= 2:
        #print("MI values:", pair["mi"].values)
        #print(f"{metric} values:", pair[metric].values)
        r, p = pearsonr(pair["mi"].astype(float), pair[metric].astype(float))

        fig, ax = plt.subplots(figsize=(14, 10))

        #scatterplot of best MI values
        ax.scatter(
            pair["mi"].values,
            pair[metric].values,
        )
        for name, x, y in zip(pair["model"], pair["mi"], pair[metric]):
            ax.annotate(
                str(name),
                (x, y),
                xytext=(3, 3), textcoords="offset points",
                fontsize=13, alpha=0.85
            )
        ax.set_xlim(0, 3.0)
        ax.set_ylim(0, 0.45)
        ax.set_xlabel("best MI values model (bits)", fontsize=15)
        ax.set_ylabel(f"brainscore {metric}", fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_title(f"correlation best MI per {metric}", fontsize=15)

        textstr = f"Pearson r = {r:.3f}\np-value = {p:.2e}\nN = {len(pair)}"
        ax.text(
            0.02, 0.02, textstr,
            transform=ax.transAxes, va="bottom", ha="left", fontsize=13,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.8)
        )

        fig.tight_layout()

    out_png = os.path.join(plot_path, f"correlation_mi_{metric}_k47.png")
    plt.savefig(out_png)
    plt.close(fig)

    rows.append({"metric": metric, "pearson_r": r, "n": n_values, "p_value": p})

corr_df = pd.DataFrame(rows)#.sort_values("metric").reset_index(drop=True)
corr_df.to_csv(os.path.join(scores_path, f"mi_brainscores_corr_{subset}_k47.csv"), index=False)
#print(corr_df)

model_to_target = {
    "alexnet":     "alexnet",
    "densenet121": "densenet-121",
    "densenet169": "densenet-169",
    "densenet201": "densenet-201",
    "googlenet":   "inception_v1",
    "inceptionv3": "inception_v3",
    "resnet101":   "resnet_101_v1",
    "resnet151":   "resnet_152_v1", 
    "resnet18":    "resnet-18",
    "resnet34":    "resnet-34",
    "resnet50":    "resnet_50_v1",
    "mobilenet":   "mobilenet_v2_1-4_224_pytorch",
    "vgg16":       "vgg_16",
    "vgg19":       "vgg_19",
}

df_mi["target_name"] = df_mi["model"].map(model_to_target)
df_mi = df_mi.copy()
combined_df = df_mi.merge(out_best, left_on="target_name", right_on=model_col, how="inner")
#print(combined_df)
combined_df.drop(columns=["target_name", "Model"], inplace=True, errors="ignore")
#print(combined_df)

for metric in brainscore_values:
    pair = combined_df[["model", "mi", metric]].dropna(subset=["mi", metric])
    if pair.empty:
        continue

    models = sorted(pair["model"].unique())
    cmap = plt.colormaps["tab20"]
    color_map = {m: cmap(i % cmap.N) for i, m in enumerate(models)}

    fig, ax = plt.subplots(figsize=(11, 7))

    for m in models:
        pm = pair[pair["model"] == m]
        ax.scatter(pm["mi"].values, pm[metric].values,
                   s=30, alpha=0.9, color=color_map[m], label=m)

    ax.set_xlim(0, 3.0)
    ax.set_xlabel("MI (bits)", fontsize=15)
    ax.set_ylabel(f"{metric}", fontsize=15)
    ax.set_title(f"All layers: MI vs {metric}", fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.legend(loc="best", fontsize=12, frameon=True, framealpha=0.8)
    fig.tight_layout()

    out_png = os.path.join(plot_path, f"correlation_allmi_{metric}_k47.png")
    plt.savefig(out_png)
    plt.close(fig)
