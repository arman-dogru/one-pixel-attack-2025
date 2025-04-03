import shutil
from scipy import stats
from typing import List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os

from main import CLASS_NAMES


def rqs(results_table: pd.DataFrame, base_dir: str):
    # ======================= RQ4 Analysis: One-Pixel Attack Consistency Across Groups =======================
    # Group results by model and class (true label) to analyze success rate
    grouped_results = results_table.groupby(['model', 'true'])
    # Calculate success rate for each combination of model and class
    success_rate_by_group = grouped_results['success'].mean().reset_index()
    # Print the success rates for each group
    print("\nSuccess rate by model and class:")
    print(success_rate_by_group)

    # Further analysis: Check if specific classes are more vulnerable across all models
    class_success_rate = results_table.groupby('true')['success'].mean().reset_index()
    # Print the success rate per class (across all models)
    print("\nSuccess rate by class (across all models):")
    print(class_success_rate)

    # ======================= Bias Investigation: Misclassified Images by Class =======================
    misclassified_by_class = results_table[results_table['success'] == False].groupby('true').size()
    # Print out the misclassified count per class
    print("\nMisclassified images by class:")
    print(misclassified_by_class)

    # Plot Misclassified Images by Class (Bar Plot)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=misclassified_by_class.index, y=misclassified_by_class.values, palette="viridis")
    plt.title('Misclassified Images by Class')
    plt.ylabel('Number of Misclassified Images')
    plt.xlabel('Class')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{base_dir}/misclassified_by_class.png", dpi=400)

    # ======================= Analyze the Impact of Different Pixel Modifications Across Classes =======================
    pixel_success_rate = results_table.groupby('pixels')['success'].mean().reset_index()
    # Print the success rate for different pixel perturbations
    print("\nSuccess rate by number of pixels perturbed:")
    print(pixel_success_rate)

    # Plot success rate by number of pixels perturbed (Bar Plot)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=pixel_success_rate, x='pixels', y='success')
    plt.title('Success Rate of One-Pixel Attacks by Number of Pixels Perturbed')
    plt.ylabel('Success Rate')
    plt.xlabel('Number of Pixels Perturbed')
    plt.tight_layout()
    plt.savefig(f"{base_dir}/success_rate_by_pixels.png", dpi=400)

    # Plot success rate by model and class (Bar Plot)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=success_rate_by_group, x='model', y='success', hue='true')
    plt.title('Success Rate of One-Pixel Attacks by Model and Class')
    plt.ylabel('Success Rate')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.legend(title="True Label", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{base_dir}/success_rate_by_model_and_class.png", dpi=400)

    # Plot success rate by class (Bar Plot)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=class_success_rate, x='true', y='success')

    plt.title('Success Rate of One-Pixel Attacks by Class')
    plt.ylabel('Success Rate')
    plt.xlabel('Class')
    plt.xticks(rotation=45)

    # Ensure the same scale for y-axis across different plots
    plt.ylim(0, 1)  # Assuming success rate is between 0 and 1

    plt.tight_layout()
    plt.savefig(f"{base_dir}/success_rate_by_class.png", dpi=400)

    # Save the success rate by group and per class to CSV for future reference
    success_rate_by_group.to_csv(f"{base_dir}/success_rate_by_group.csv", index=False)
    class_success_rate.to_csv(f"{base_dir}/class_success_rate.csv", index=False)


    # --- Chi-Squared Test for Independence ---
    # Create a contingency table for 'model' vs 'success'
    contingency_table = pd.crosstab(results_table['model'], results_table['success'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    print("\nChi-Squared Test for independence between model and attack success:")
    print(f"Chi2 statistic: {chi2}, p-value: {p_value}, degrees of freedom: {dof}")

    # --- One-Way ANOVA across classes ---
    groups = [group["success"].values for name, group in results_table.groupby("true")]
    f_stat, anova_p = stats.f_oneway(*groups)
    print("\nOne-Way ANOVA across classes for attack success rates:")
    print(f"F-statistic: {f_stat}, p-value: {anova_p}")

    # --- Heatmap Visualization of Success Rate by Model and Class ---
    heatmap_data = success_rate_by_group.pivot(index='model', columns='true', values='success')
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="viridis", cbar=True)
    plt.title("Heatmap: Success Rate of One-Pixel Attacks by Model and Class")
    plt.xlabel("True Class")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(f"{base_dir}/success_rate_heatmap.png", dpi=400)


    # Function to visualize original and perturbed images side by side
    def visualize_image_comparison(original_image, perturbed_image, true_label):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(original_image)
        ax[0].set_title(f"Original: {CLASS_NAMES[true_label]}")
        ax[0].axis('off')
        ax[1].imshow(perturbed_image)
        ax[1].set_title(f"Perturbed: {CLASS_NAMES[true_label]}")
        ax[1].axis('off')
        plt.tight_layout()


    # Display side-by-side comparisons for a few successful attacks
    sample_successes = results_table[results_table['success'] == True].sample(n=5, random_state=42)
    for index, row in sample_successes.iterrows():
        visualize_image_comparison(row["original_image"], row["attack_image"], row["true"])
        os.makedirs(f"{base_dir}/image_comparisons", exist_ok=True)
        plt.savefig(f"{base_dir}/image_comparisons/{index}.png", dpi=400)
        plt.close()

    # Plot the mean change in pixels for successful attacks by model and pixel count in a heatmap of the most changed pixel where each image is (32,32,3)
    plt.figure(figsize=(10, 5))
    data = pd.DataFrame(baseline[baseline["success"]])
    sns.heatmap(data.groupby(["model", "pixels"])["diff_image"].mean().apply(lambda x: x.max()).unstack())
    plt.savefig(f"{base_dir}/pixel_heatmap.png", dpi=400)
    plt.close()


def find_overlap_images(configs: List[Tuple[str, str]]):

    # Load each dataframe and stack them
    dataframes = []
    for input, results_name in configs:
        df = pd.read_pickle(input)
        df["results_name"] = results_name
        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)
    df = df[df["model"] == "resnet"]

    # Get all the images that have been reported for all models and all results_name
    images = df.groupby(["pixels", "model", "image"])

    # Count the overlap between each result_name
    print(df["image"].value_counts())
    print(df[df["image"] == 8549])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analysis Options")
    parser.add_argument("--dev", action="store_true", help="Development Mode")
    parser.add_argument("--images", action="store_true", help="Save Image Pairs")
    parser.add_argument("--rqs", action="store_true", help="Run RQ4 Analysis")
    args = parser.parse_args()

    configs = [
        ("/kaggle/working/baseline_results.pkl", "baseline_results"),
        # ('results/baseline.pkl', 'baseline'),
        ("/kaggle/working/defense.py", "defense_all"),
        ("/kaggle/working/defense_results_blur.pkl", "defense_blur"),
        ("/kaggle/working/defense_results_noise.pkl", "defense_noise"),
        ("/kaggle/working/defense_results_simclr.pkl", "defense_simclr"),
        ("/kaggle/working/minimize_change_resnet.pkl", "minimize_change_resnet"),
        # ('results/minimize_change.pkl', 'minimize_change'),
    ]

    for input, results_name in configs:
        base_dir = f"results/{results_name}"
        os.makedirs(base_dir, exist_ok=True)
        baseline = pd.read_pickle(input)
        baseline = baseline[baseline["model"] != "lenet"]
        baseline["diff_image"] = baseline["original_image"] - baseline["attack_image"]
        baseline["model"] = baseline["model"].map(
            {
                "resnet": "ResNet",
                "net_in_net": "Net in Net",
                "densenet": "DenseNet",
                "pure_cnn": "Pure CNN",
                "wide_resnet": "Wide ResNet",
            }
        )

        # Group by model and pixels and plot the mean of success on heatmap
        plt.figure(figsize=(8, 5))
        sns.heatmap(
            baseline.groupby(["model", "pixels"])["success"].mean().unstack() * 100,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
        )
        plt.title("Mean Success Rate (%)")
        plt.xlabel("Changed Pixel Count")
        plt.ylabel("Model")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{base_dir}/heatmap.png", dpi=400)
        plt.close()

        plt.figure(figsize=(8, 5))
        sns.barplot(
            data=baseline[baseline["model"] == "ResNet"],
            x="pixels",
            y="success",
            hue="model",
        )
        plt.title("Mean Success Rate (%)")
        plt.xlabel("Changed Pixel Count")
        plt.ylabel("Success Rate")
        plt.xticks(rotation=45)
        plt.savefig(f"{base_dir}/barplot.png", dpi=400)
        plt.close()

        if args.rqs:
            rqs(baseline, base_dir)

        if args.images:
            images = [baseline[baseline["success"]].index[0]]
            if os.path.exists(f"{base_dir}/image_pairs"):
                shutil.rmtree(f"{base_dir}/image_pairs")
            os.makedirs(f"{base_dir}/image_pairs", exist_ok=True)

            for image_id in images:
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0].imshow(baseline["original_image"][image_id])
                axs[0].set_title("Original Image")
                axs[1].imshow(baseline["attack_image"][image_id])
                axs[1].set_title("Attack Image")
                axs[0].axis("off")
                axs[1].axis("off")
                plt.tight_layout()

                plt.savefig(f"{base_dir}/image_pairs/{image_id}.png", dpi=400)

    # Compare the defense results
    result_files = [
        "results/defense_results_all.pkl",
        "results/defense_results_blur.pkl",
        "results/defense_results_noise.pkl",
        "results/defense_results_simclr.pkl",
    ]

    # Load the dfs and stack them
    dataframes = []
    for file in result_files:
        df = pd.read_pickle(file)
        label = file.split("_")[-1].split(".")[0]
        df["defense"] = label
        df["defense"] = df["defense"].map(
            {
                "all": "All",
                "blur": "Blur",
                "noise": "Noise",
                "simclr": "SimCLR",
            }
        )
        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)
    baseline = pd.read_pickle("/kaggle/working/baseline_results.pkl")
    baseline = baseline[baseline["model"] == "resnet"]
    baseline['defense'] = "No Defense"
    df = pd.concat([df, baseline], ignore_index=True)
    df['success'] = df['success'].astype(float) * 100

    # Group by model and pixels and plot the mean of success on heatmap
    plt.figure(figsize=(8, 5))
    sns.heatmap(
        df.groupby(["defense", "pixels"])["success"].mean().unstack(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={"label": "Mean Success Rate (%)"},
    )
    plt.xlabel("Changed Pixel Count")
    plt.ylabel("Defense Strategy")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"results/defense_heatmap.png", dpi=400)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df[df["model"] == "resnet"],
        x="pixels",
        y="success",
        errorbar='se',
        hue="defense",
    )
    plt.xlabel("Changed Pixel Count")
    plt.ylabel("Success Rate (%)")
    plt.legend(loc="upper left", title="Defense Strategy")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"results/defense_barplot.png", dpi=400)


    # Compare the minimize change results
    # baseline = pd.read_pickle("results/minimize_change_resnet.pkl")
    baseline = pd.read_pickle("results/minimize_local_resnet_small.pkl")
    columns = pd.Index(
        [
            "model",
            "attack_image",
            "original_image",
            "pixels",
            "image",
            "true",
            "predicted",
            "success",
            "cdiff",
            "prior_probs",
            "predicted_probs",
            "perturbation",
        ]
    )
    baseline = pd.DataFrame(baseline, columns=columns)
    baseline = baseline[baseline["model"] == "resnet"]
    baseline = baseline[baseline["success"]]

    patch_size = 3  # Size of the patch to be used  nxn, can also be 5x5 or 7x7
    minus = patch_size // 2
    plus = patch_size - minus
    changes = []
    diffs = []
    for original, attack_image in zip(baseline["original_image"].values, baseline["attack_image"].values):
        diff = np.abs(original - attack_image)
        diff = np.pad(diff, ((minus, minus), (minus, minus), (0, 0)), mode='constant')  # Handle borders [9]
        y_list, x_list = np.array(np.argwhere(diff)[:, :2][::3]).T

        differences_from_patch = []
        for y, x in zip(y_list, x_list):
            patch = np.array(attack_image[y - minus:y + plus, x - minus:x + plus, :])
            if patch.shape != (patch_size, patch_size, 3):
                continue
            mask = np.ones_like(patch, dtype=bool)
            mask[minus, minus, :] = False
            masked_patch = np.where(mask, patch, np.nan)
            avg = np.nanmean(masked_patch, axis=(0, 1))

            # Euclidean dist from original
            dist = np.linalg.norm(avg - patch[minus, minus, :])
            differences_from_patch.append(dist)

        # Normalize change to the size of the pixel (0-255)
        change = np.array(differences_from_patch) / 255
        changes.append(np.mean(change))
        diffs.append(diff)

    changes = np.array(changes)
    diffs = np.array(diffs)

    mean_diffs = np.mean(diffs, axis=0)

    # Plot mean_diffs as an image (32, 32, 3)
    plt.figure(figsize=(5, 5))
    plt.imshow(mean_diffs)
    plt.axis("off")
    plt.title("Mean Change in Pixels Across RGB Channels")
    plt.tight_layout()
    plt.savefig(f"results/minimize_change_resnet_mean_diff.png", dpi=400)
    plt.close()

    # Plot histogram of changes
    plt.figure(figsize=(8, 5))
    plt.hist(changes * 255, bins=50)
    plt.xlabel("Euclidean Distance From Surrounding 3x3 Pixel Cluster Mean (0-255)")
    plt.ylabel("Frequency (50 Bins)")
    plt.tight_layout()
    plt.savefig(f"results/minimize_change_resnet_histogram.png", dpi=400)
    plt.close()

    # Plot an example diff image
    before = baseline["original_image"].values[0]
    after = baseline["attack_image"].values[0]
    diff = np.abs(before - after)

    # Plot the before and after images
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(before)
    axs[0].set_title("Original Image")
    axs[1].imshow(after)
    axs[1].set_title("Attack Image")
    axs[0].axis("off")
    axs[1].axis("off")
    plt.tight_layout()
    plt.savefig(f"results/minimize_change_resnet_example.png", dpi=400)
    plt.close()

    # Plot the diff image
    plt.figure(figsize=(5, 5))
    plt.imshow(diff)
    plt.axis("off")
    plt.title("Difference Image")
    plt.tight_layout()
    plt.savefig(f"results/minimize_change_resnet_diff_example.png", dpi=400)
    plt.close()


    print('Pause')

