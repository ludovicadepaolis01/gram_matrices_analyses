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
import matplotlib.cm as cm
import matplotlib.colors as mcolors

subset = "all"
data_path = f"/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/csvs/mi_csv_subset_{subset}_k47.csv"
brainscore_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/csvs/leaderboard.csv"
scores_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses"
plot_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/plots"

df_mi = pd.read_csv(data_path)

#model mask
pretty_model_names = {
    "alexnet":      "AlexNet",
    "densenet121":  "DenseNet-121",
    "densenet169":  "DenseNet-169",
    "densenet201":  "DenseNet-201",
    "inceptionv3":  "InceptionV3",
    "mobilenet":    "MobileNetV2",
    "resnet18":     "ResNet18",
    "resnet34":     "ResNet34",
    "resnet50":     "ResNet50",
    "resnet101":    "ResNet101",
    "resnet152":    "ResNet152",
    "vgg16":        "VGG-16",
    "vgg19":        "VGG-19",
}

#brainscore mask
targets = ["alexnet", "densenet-121", "densenet-169", 
           "densenet-201", "inception_v3", 
           "resnet_101_v1", "resnet_152_v1", "resnet-18",
           "resnet-34", "resnet_50_v1", "mobilenet_v2_1-4_224_pytorch",
           "vgg_16", "vgg_19"]

pretty_metric_names = {
    "average_vision":  "Average Vision",
    "neural_vision":   "Neural Vision",
    "behavior_vision": "Behavior Vision",
    "v1":              "V1",
    "v2":              "V2",
    "v4":              "V4",
    "it":              "IT",
}

#extract top MI per model per layer
top_mi = df_mi.loc[df_mi.groupby("model")["mi"].idxmax(), ["model","layer","mi"]].reset_index(drop=True)
#drop models not present in brainscore
top_mi = top_mi[top_mi["model"] != "googlenet"]
top_mi = top_mi.reset_index(drop=True)
print(top_mi)

#open brainscore leaderboard which is a bit problematic
df_brainscore = pd.read_csv(
                brainscore_path,
                comment="#", #drops useless line
                sep=None, #auto-detects comma vs tab
                engine="python",
                encoding="utf-8-sig",
                na_values=["", "NaN", "nan", "—", "–"]  #treat dashes as NaN
)

#clean column names
df_brainscore.columns = df_brainscore.columns.str.strip()

# 3) Make everything numeric except 'Model' (and any other obvious text cols)
for col in df_brainscore.columns:
    if col.lower() not in {"model"}:
        #remove any stray non-numeric chars (NBSPs, etc.) before conversion
        df_brainscore[col] = pd.to_numeric(
            df_brainscore[col].astype(str).str.replace(r"[^\d\.\-eE]", "", regex=True),
            errors="coerce"
        )

model_col = next(c for c in df_brainscore.columns if c.strip().lower() in {"model", "model_name", "name"})

#Normalize column names to match keys like "average_vision"
norm = {c: c.strip().lower().replace(" ", "_") for c in df_brainscore.columns}
#print(norm)
inv_norm = {v: k for k, v in norm.items()}
#print(inv_norm)

wanted_keys = ["average_vision", "neural_vision", "behavior_vision", "v1", "v2", "v4", "it"]
wanted_cols = [inv_norm[k] for k in wanted_keys if k in inv_norm]

# Filter the two models (case-insensitive, exact match after strip)
mask = df_brainscore[model_col].astype(str).str.strip().str.lower().isin([t.lower() for t in targets])

out = df_brainscore.loc[mask, [model_col] + wanted_cols].copy()
#print(out)

#Nice, consistent column names
rename_map = {inv_norm[k]: k for k in wanted_keys if k in inv_norm}
out = out.rename(columns=rename_map)
#print(out)

#if there are duplicate models, select only the one with the highest "average_vision" score
out_best = out.loc[out.groupby(model_col)["average_vision"].idxmax()].reset_index(drop=True)
scores = out_best.set_index(model_col).to_dict(orient="index")
out_best.to_csv(os.path.join(scores_path, "best_brainscores.csv"), index=False)

#calculate pearsons r among MI values and the selected brainscores values
combined_df = pd.concat(
    [top_mi.drop(columns=["layer"]).reset_index(drop=True),
    out_best.drop(columns=["Model"]).reset_index(drop=True)],
    axis=1
)

#columns of brainscore values to correlate against mi 
brainscore_values = [c for c in combined_df.columns if c in {"average_vision", "neural_vision", "behavior_vision", "v1", "v2", "v4", "it"}]

mi_all = pd.to_numeric(combined_df["mi"], errors="coerce")
model_all = combined_df["model"].astype(str)
valid = mi_all.notna()
tick_pos = mi_all[valid].values
tick_labels = [f"{x:.3g}\n{m}" for x, m in zip(tick_pos, model_all[valid])]

rows = []

n_metrics = len(brainscore_values)
n_cols = 4
n_rows = int(np.ceil(n_metrics / n_cols))

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(4 * n_cols, 4 * n_rows),
    constrained_layout=True,
)
axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

# shared models + colors
all_models = sorted(combined_df["model"].astype(str).unique())
cmap = cm.get_cmap("tab20", len(all_models))
model_to_color = {m: cmap(i) for i, m in enumerate(all_models)}

for i_metric, metric in enumerate(brainscore_values):
    pretty_metric = pretty_metric_names.get(metric, metric)

    row = i_metric // n_cols
    col = i_metric % n_cols
    ax = axes[row, col]

    brain_score = pd.to_numeric(combined_df[metric], errors="coerce")
    mi_score = pd.to_numeric(combined_df["mi"], errors="coerce")
    pair = pd.DataFrame(
        {"model": combined_df["model"], "mi": mi_score, metric: brain_score}
    ).dropna(subset=["mi", metric])

    n_values = len(pair)
    if n_values < 2:
        ax.set_visible(False)
        continue

    r, p = pearsonr(pair["mi"].astype(float), pair[metric].astype(float))

    # scatter...
    for model in all_models:
        mask = (pair["model"].astype(str) == model)
        if not mask.any():
            continue
        ax.scatter(
            pair.loc[mask, "mi"].values,
            pair.loc[mask, metric].values,
            color=model_to_color[model],
            s=80,
        )

    ax.set_xlim(0, 3.0)
    ax.set_ylim(0, 0.45)
    ax.tick_params(axis="both", which="major", labelsize=10)

    # Title at the top
    ax.set_title(pretty_metric, fontsize=12, pad=8)

    # ---- AXIS LABELS ONLY ON SELECTED SUBPLOTS ----
    # y-label on "Average Vision" and "V2"
    if pretty_metric in ["Average Vision", "V2"]:
        ax.set_ylabel("Brainscore", fontsize=14)

    # x-label on "V2", "V4", and "IT"
    if pretty_metric in ["V2", "V4", "IT"]:
        ax.set_xlabel("MI value (bits)", fontsize=14)

    # correlation box
    textstr = f"Pearson r = {r:.3f}\np-value = {p:.2e}\nN = {n_values}"
    ax.text(
        0.02, 0.02, textstr,
        transform=ax.transAxes,
        va="bottom", ha="left", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white",
                  edgecolor="black", alpha=0.8),
    )

    rows.append({"metric": metric, "pearson_r": r, "n": n_values, "p_value": p})

# hide unused axes
for j in range(i_metric + 1, n_rows * n_cols):
    row = j // n_cols
    col = j % n_cols
    axes[row, col].set_visible(False)


handles = [
    plt.Line2D([], [], marker="o", linestyle="",
               markersize=8, color=model_to_color[m])
    for m in all_models
]
labels = [pretty_model_names.get(m, m) for m in all_models]

fig.legend(
    handles, labels,
    title="Model",
    loc="center right",          # attach to the left side of the anchor
    bbox_to_anchor=(0.8, 0.3), # (x,y) in figure coords: right side, middle
    borderaxespad=0.,
    fontsize=10,
    title_fontsize=11,
)

# leave room on the right for the legend (xmax < 1)
plt.tight_layout(rect=[0.06, 0.06, 0.85, 0.95])
out_png = os.path.join(plot_path, "correlation_mi_all_brainscores_k47.png")
plt.savefig(out_png)
plt.close(fig)

corr_df = pd.DataFrame(rows)
corr_df.to_csv(os.path.join(scores_path, f"mi_brainscores_corr_{subset}_k47.csv"),
               index=False)

'''
rows = []
for index, metric in enumerate(brainscore_values):

    pretty_metric = pretty_metric_names.get(metric, metric)
    brain_score = pd.to_numeric(combined_df[metric], errors="coerce")
    mi_score = pd.to_numeric(combined_df["mi"], errors="coerce")
    pair = pd.DataFrame({"model": combined_df["model"],
                        "mi": mi_score, 
                        metric: brain_score})
    n_values = len(pair)
    if n_values >= 2:

        r, p = pearsonr(pair["mi"].astype(float), pair[metric].astype(float))

        fig, ax = plt.subplots(figsize=(14, 10),
                               constrained_layout=True,)
        models = sorted(pair["model"].astype(str).unique())
        colors = cm.get_cmap("tab20", len(models))

        for index, model in enumerate(models):
            mask = (pair["model"].astype(str) == model)
            pretty_model = pretty_model_names.get(model, model)
            #scatterplot of best MI values
            ax.scatter(
                pair.loc[mask, "mi"].values,
                pair.loc[mask, metric].values,
                color=colors(index),
                label=pretty_model, #one legend entry per model
                s=80, #point size
            )
            for x, y in zip(pair.loc[mask, "mi"], pair.loc[mask, metric]):
                ax.annotate(
                    pretty_model,
                    (x, y),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=15
                    )
            ax.set_xlim(0, 3.0)
            ax.set_ylim(0, 0.45)
            ax.set_xlabel("MI value (bits)", fontsize=15)
            ax.set_ylabel(f"Brainscore {pretty_metric}", fontsize=15)
            ax.tick_params(axis='both', which='major', labelsize=15)
            #ax.set_title(f"correlation best MI per {metric}", fontsize=15)
            leg = ax.legend(title="Model", loc="upper left", fontsize=15)
            leg.get_title().set_fontsize(15)  # title font size

            textstr = f"Pearson r = {r:.3f}\np-value = {p:.2e}\nN = {len(pair)}"
            ax.text(
                0.02, 0.02, textstr,
                transform=ax.transAxes, va="bottom", ha="left", fontsize=13,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.8)
            )

    out_png = os.path.join(plot_path, f"correlation_mi_{metric}_k47.png")
    plt.savefig(out_png)
    plt.close(fig)

    rows.append({"metric": metric, "pearson_r": r, "n": n_values, "p_value": p})

corr_df = pd.DataFrame(rows)#.sort_values("metric").reset_index(drop=True)
corr_df.to_csv(os.path.join(scores_path, f"mi_brainscores_corr_{subset}_k47.csv"), index=False)
'''
'''
model_to_target = {
    "alexnet":     "alexnet",
    "densenet121": "densenet-121",
    "densenet169": "densenet-169",
    "densenet201": "densenet-201",
    "inceptionv3": "inception_v3",
    "mobilenet":   "mobilenet_v2_1-4_224_pytorch",
    "resnet18":    "resnet-18",
    "resnet34":    "resnet-34",
    "resnet50":    "resnet_50_v1",
    "resnet101":   "resnet_101_v1",
    "resnet152":   "resnet_152_v1", 
    "vgg16":       "vgg_16",
    "vgg19":       "vgg_19",
}

df_mi["target_name"] = df_mi["model"].map(model_to_target)
df_mi = df_mi.copy()
combined_df = df_mi.merge(out_best, left_on="target_name", right_on=model_col, how="inner")
combined_df.drop(columns=["target_name", "Model"], inplace=True, errors="ignore")

for metric in brainscore_values:
    pair = combined_df[["model", "mi", metric]].dropna(subset=["mi", metric])
    if pair.empty:
        continue

    fig, ax = plt.subplots(figsize=(11, 7), 
                           constrained_layout=True,)

    models = sorted(pair["model"].unique())
    colors = cm.get_cmap("tab20", len(models))

    for index, model in enumerate(models):
        pm = pair[pair["model"] == model]
        display_name = pretty_model_names.get(model, model)
        ax.scatter(pm["mi"].values, pm[metric].values,
                   s=30, alpha=0.9, color=colors(index), label=display_name)

    ax.set_xlim(0, 3.0)
    ax.set_xlabel("MI (bits)", fontsize=15)
    ax.set_ylabel(f"{pretty_metric}", fontsize=15)
    #ax.set_title(f"All layers: MI vs {metric}", fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    leg = ax.legend(title="Model", loc="upper left", fontsize=12, frameon=True, framealpha=0.8)
    leg.get_title().set_fontsize(15)  # title font size
    out_png = os.path.join(plot_path, f"correlation_allmi_{metric}_k47.png")
    plt.savefig(out_png)
    plt.close(fig)
'''