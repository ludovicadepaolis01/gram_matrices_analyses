import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = "/leonardo/home/userexternal/ldepaoli/lab/gram_project/vgg_analyses/info_plots_s30000_20052025/" 
save_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_project/vgg_analyses/plots/vgg16_losses_textures.png"
print(os.listdir(path))
all_dfs = []

for filename in os.listdir(path):
    if filename.endswith(".csv"):
        file_path = os.path.join(path, filename)
        df = pd.read_csv(file_path, index_col=0)
        df = df.iloc[-30000:]
        all_dfs.append(df)

combined_df = pd.concat(all_dfs).reset_index()
combined_df["optim_step"] = combined_df["optim_step"].astype(int)

sorted_textures = (
combined_df.groupby("texture")["loss"]
.mean()
.sort_values()
.index.tolist()
)

palette = sns.color_palette("Blues", n_colors=len(sorted_textures))

plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

for i, texture in enumerate(sorted_textures):
    texture_df = combined_df[combined_df["texture"] == texture]
    sns.lineplot(
    data=texture_df,
    x="optim_step",
    y="loss",
    label=texture,
    color=palette[i]
    )

plt.legend(title="texture labels", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.title("textures generation losses with vgg16")
plt.xlabel("optim_step")
plt.ylabel("loss")
plt.tight_layout()
plt.savefig(save_path, bbox_inches = "tight", pad_inches=0.1)
plt.show()