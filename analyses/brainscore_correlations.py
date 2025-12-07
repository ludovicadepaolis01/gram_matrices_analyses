import os
import numpy as np 
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib as mpl

#input paths
mi_data_path = f"/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/csvs_debug/mi_csv_k47.csv"
brainscore_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/csvs/leaderboard.csv"

#output paths
scores_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses"
plot_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/plots"

#masks
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

targets = ["alexnet", 
           "densenet-121", 
           "densenet-169", 
           "densenet-201", 
           "inception_v3", 
           "mobilenet_v2_1-4_224_pytorch",
           "resnet_101_v1", 
           "resnet_152_v1", 
           "resnet-18",
           "resnet-34", 
           "resnet_50_v1", 
           "vgg_16", 
           "vgg_19",]

pretty_metric_names = {
    "average_vision":  "Average Vision",
    "neural_vision":   "Neural Vision",
    "behavior_vision": "Behavior Vision",
    "v1":              "V1",
    "v2":              "V2",
    "v4":              "V4",
    "it":              "IT",
}

def brainscore_corr(
        mi_data_path,
        brainscore_path,
        scores_path,
        plot=True,
        plot_path=".",
    ):

    df_mi = pd.read_csv(mi_data_path)
    #print(df_mi["model"].unique())

    #extract top MI per model per layer
    top_mi = df_mi.loc[df_mi.groupby("model")["mi"].idxmax(), ["model","layer","mi"]].reset_index(drop=True)
    top_mi = top_mi.reset_index(drop=True)
    #print(top_mi)

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

    # Make everything numeric except 'Model' (and any other obvious text cols)
    for col in df_brainscore.columns:
        if col.lower() not in {"model"}:
            #remove weird characterds
            df_brainscore[col] = pd.to_numeric(
                df_brainscore[col].astype(str).str.replace(r"[^\d\.\-eE]", "", regex=True),
                errors="coerce"
            )

    model_col = next(c for c in df_brainscore.columns if c.strip().lower() in {"model", "model_name", "name"})

    #normalize column names to match keys like "average_vision"
    norm = {c: c.strip().lower().replace(" ", "_") for c in df_brainscore.columns}
    inv_norm = {v: k for k, v in norm.items()}

    wanted_keys = ["average_vision", "neural_vision", "behavior_vision", "v1", "v2", "v4", "it"]
    wanted_cols = [inv_norm[k] for k in wanted_keys if k in inv_norm]

    mask = df_brainscore[model_col].astype(str).str.strip().str.lower().isin([t.lower() for t in targets])

    out = df_brainscore.loc[mask, [model_col] + wanted_cols].copy()

    rename_map = {inv_norm[k]: k for k in wanted_keys if k in inv_norm}
    out = out.rename(columns=rename_map)

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

    '''
    #optionally check if the models are aligned!
    print("=== MI top models ===")
    print(top_mi["model"].reset_index(drop=True))

    print("\n=== Brainscore models ===")
    print(out_best[model_col].reset_index(drop=True))
    '''

    #columns of brainscore values to correlate against mi 
    brainscore_values = [c for c in combined_df.columns if c in {"average_vision", "neural_vision", "behavior_vision", "v1", "v2", "v4", "it"}]
    
    '''
    #optionally add models labels to datapoint
    mi_all = pd.to_numeric(combined_df["mi"], errors="coerce")
    model_all = combined_df["model"].astype(str)
    valid = mi_all.notna()
    tick_pos = mi_all[valid].values
    tick_labels = [f"{x:.3g}\n{m}" for x, m in zip(tick_pos, model_all[valid])]
    '''
    
    rows = []

    n_metrics = len(brainscore_values)
    n_cols = 4
    n_rows = int(np.ceil(n_metrics / n_cols))

    if plot:
        #plot one .png with as many subplots as brainscore values
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4 * n_cols, 4 * n_rows),
        )

        axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

        # shared models + colors
        all_models = sorted(combined_df["model"].astype(str).unique())
        cmap = mpl.colormaps.get_cmap("tab20")
        n = max(1, len(all_models) - 1)
        model_to_color =  {m: cmap(i / n) for i, m in enumerate(all_models)}

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
            ax.tick_params(axis="both", which="major", labelsize=15)
            ax.set_title(pretty_metric, fontsize=15, pad=8)

            # y-label on "Average Vision" and "V2"
            if pretty_metric in ["Average Vision", "V2"]:
                ax.set_ylabel("Brainscore", fontsize=15)

            # x-label on "V2", "V4", and "IT"
            if pretty_metric in ["V2", "V4", "IT"]:
                ax.set_xlabel("MI value (bits)", fontsize=15)

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
            loc="center right",
            bbox_to_anchor=(0.8, 0.28),
            fontsize=12,
            title_fontsize=15,
        )

        # leave room on the right for the legend
        plt.tight_layout(rect=[0.06, 0.06, 0.85, 0.95])
        out_png = os.path.join(plot_path, "correlation_mi_all_brainscores_k47.png")
        plt.savefig(out_png)
        plt.close(fig)

        corr_df = pd.DataFrame(rows)
        corr_df.to_csv(os.path.join(scores_path, f"mi_brainscores_corr_k47.csv"),
                    index=False)

brainscore_corr(
    mi_data_path=mi_data_path,
    brainscore_path=brainscore_path,
    scores_path=scores_path,
    plot_path=plot_path,
)