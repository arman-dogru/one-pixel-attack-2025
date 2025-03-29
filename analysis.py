import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analysis Options")
    parser.add_argument("--dev", action="store_true", help="Development mode")
    args = parser.parse_args()

    configs = [
        ('results/baseline_results.pkl', 'baseline'),
        ('results/defense_results_all.pkl', 'defense_all'),

    ]
    for input, results_name in configs:
        base_dir = f"results/{results_name}"
        os.makedirs(base_dir, exist_ok=True)
        baseline = pd.read_pickle(input)
        baseline["diff_image"] = baseline["original_image"] - baseline["attack_image"]

        # Group by model and pixels and plot the mean of success on heatmap
        plt.figure(figsize=(10, 5))
        sns.heatmap(
            baseline.groupby(["model", "pixels"])["success"].mean().unstack() * 100,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
        )
        plt.title("Mean Success Rate (%)")
        plt.savefig(f"{base_dir}/heatmap.png", dpi=400)

        # Plot the mean change in pixels for successful attacks by model and pixel count in a heatmap of the most changed pixel where each image is (32,32,3)
        plt.figure(figsize=(10, 5))
        data = pd.DataFrame(baseline[baseline["success"]])

        sns.heatmap(
            data.groupby(["model", "pixels"])["diff_image"]
            .mean()
            .apply(lambda x: x.max())
            .unstack()
        )

        plt.savefig(f"{base_dir}/pixel_heatmap.png", dpi=400)
        data_generator = baseline.groupby(
            [
                "model",
                "pixels",
                "success",
            ]
        )

        if args.dev:
            for (model, pixel_count, success), value in data_generator["diff_image"]: #type: ignore
                os.makedirs(f"{base_dir}/average_maximum_changes", exist_ok=True)
                average_changes = np.array(value.abs().mean()).max(
                    axis=2
                )  # Average over the 3 channels
                average_changes = np.log(abs(average_changes) + 1)
        
                plt.figure(figsize=(6, 5))
                sns.heatmap(average_changes, cmap="coolwarm")
                plt.title(f"Model: {model}, Pixels: {pixel_count}")
                plt.xlabel("Width")
                plt.ylabel("Height")
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                worked_label = "successful" if success else "unsuccessful"
                plt.savefig(f"{base_dir}/average_maximum_changes/{worked_label}_{model}_{pixel_count}.png", dpi=400)
                plt.close()

    

