import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

model_list = [
    "vgg16",
    "vgg19",
    "alexnet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "googlenet",
    "inceptionv3",
    "mobilenet",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201"
]

#parse command-line argument
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, choices=[model for model in model_list], 
                    help="Which model to run")
#parser.add_argument("--mode", type=str, required=True, choices=["orig", "reco"], 
#                    help="Gram matrix mode: orig or reco")
args = parser.parse_args()
model_name = args.model
#optim_steps = args.optim_steps
#mode = args.mode

path = f"/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/info_plots/info_plot_{model_name}_reco"
save_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/plots"
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

plt.title(f"{model_name} textures generation losses")
plt.xlabel("optim_step")
plt.ylabel("loss")
plt.tight_layout()
plt.savefig(os.path.join(save_path, f"{model_name}_generation_losses.png"), bbox_inches = "tight", pad_inches=0.1)
plt.close()