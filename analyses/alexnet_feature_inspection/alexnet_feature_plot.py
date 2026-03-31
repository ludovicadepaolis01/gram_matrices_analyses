import os
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

#alexnet
#params
device = "cuda" if torch.cuda.is_available() else "cpu"

alexnet_name = "alexnet"

layer_indices_vgg = [1, 8, 15, 28, 41]
layer_indices_alexnet = [0, 3, 6, 8, 10]

#paths
alexnet_features_path = f"/leonardo_work/Sis25_piasini/ldepaoli/gram_matrices_analyses/features/{alexnet_name}/dtd"
if not os.path.exists(alexnet_features_path):
    os.makedirs(alexnet_features_path, exist_ok=True)

features_root_vgg19_all = "/leonardo_scratch/fast/Sis25_piasini/ldepaoli/clip_textures/features"

vgg_roots = [
    os.path.join(features_root_vgg19_all, "vgg19")]

plot_path = "/leonardo/home/userexternal/ldepaoli/lab/gram_matrices_analyses/plots"
if not os.path.exists(plot_path):
        os.makedirs(plot_path, exist_ok=True)

def feature_plot(
        layer_indices_vgg=layer_indices_vgg,
        layer_indices_alexnet=layer_indices_alexnet,
        alexnet_name=alexnet_name,
        alexnet_features_path=alexnet_features_path,
        vgg_roots=vgg_roots,
        plot=True,
        plot_path=plot_path,      
):  
    x = np.arange(1, len(layer_indices_alexnet) + 1) #shared indexing for layers

    #alexnet
    layer_means_alexnet = []
    layer_stds_alexnet = []

    for l in sorted(layer_indices_alexnet):
        fname = f"{alexnet_name}_features_dtd_layer_{l}.pt"
        features_path = os.path.join(alexnet_features_path, fname)

        feature = torch.load(features_path, map_location="cpu").float()
        print(l, feature.shape)

        layer_mean_alexnet = feature.mean().item()
        layer_std_alexnet = feature.std().item()

        layer_means_alexnet.append(layer_mean_alexnet)
        layer_stds_alexnet.append(layer_std_alexnet)

    #vgg19
    for mr in vgg_roots:
        model_name = os.path.basename(mr)
        print(model_name)
        datasets = sorted(d for d in os.listdir(mr)
                        if os.path.isdir(os.path.join(mr, d)))

        for dataset_name in datasets:
            if dataset_name == "dtd": 

                ds_dir = os.path.join(mr, dataset_name)
                print(ds_dir)

                if model_name == "vgg19":

                    layer_means_vgg = []
                    layer_stds_vgg = []
                    for l in sorted(layer_indices_vgg):
                        fname = f"{model_name}_features_{dataset_name}_layer_{l}.pt"
                        feature_path = os.path.join(ds_dir, fname)
                        feature = torch.load(feature_path, map_location="cpu")
                        print(feature.shape)

                        # one scalar mean/std for the whole layer
                        layer_mean_vgg = feature.mean().item()
                        layer_std_vgg  = feature.std().item()

                        layer_means_vgg.append(layer_mean_vgg)
                        layer_stds_vgg.append(layer_std_vgg)

                        del feature

                    if plot:
                        plt.figure(figsize=(8, 5))

                        plt.errorbar(
                            x,
                            layer_means_alexnet,
                            yerr=layer_stds_alexnet,
                            fmt='o-',
                            capsize=5,
                            color='blue',
                            label='AlexNet'
                        )

                        plt.errorbar(
                            x,
                            layer_means_vgg,
                            yerr=layer_stds_vgg,
                            fmt='o-',
                            capsize=5,
                            color='orange',
                            label='VGG19'
                        )

                        plt.title("AlexNet vs VGG19 feature statistics on DTD")
                        plt.xlabel("Relative layer index")
                        plt.ylabel("Feature mean ± std")
                        plt.xticks(x, [1, 2, 3, 4, 5])
                        plt.legend()
                        plt.tight_layout()

                        out = os.path.join(plot_path, "alexnet_vgg19_dtd_stats.png")
                        plt.savefig(out, bbox_inches="tight")
                        plt.close()

                        print(f"saved stats plot to: {out}")


feature_plot()