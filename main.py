#!/usr/bin/env python3

import argparse

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import cifar10
import pickle
import os
from huggingface_hub import snapshot_download, login, HfApi

# Custom Networks
from networks.lenet import LeNet
from networks.pure_cnn import PureCnn
from networks.network_in_network import NetworkInNetwork
from networks.resnet import ResNet
from networks.densenet import DenseNet
from networks.wide_resnet import WideResNet
from networks.capsnet import CapsNet

# Helper functions
import helper
from attack import PixelAttacker
from defense import apply_gaussian_blur, add_gaussian_noise, simclr_augmentation


if __name__ == "__main__":
    model_defs = {
        "lenet": LeNet,
        "pure_cnn": PureCnn,
        "net_in_net": NetworkInNetwork,
        "resnet": ResNet,
        "densenet": DenseNet,
        "wide_resnet": WideResNet,
    }

    login(
        new_session=False,  # Won’t request token if one is already saved on machine
        write_permission=False,  # Requires a token with write permission
        token=os.environ["HF_TOKEN_WRITE"],
    )
    os.makedirs("networks/models", exist_ok=True)
    snapshot_download(
        repo_id="Ethics2025W/base",
        local_dir="networks/models",
    )

    parser = argparse.ArgumentParser(description="Attack models on Cifar10")
    parser.add_argument(
        "--model",
        nargs="+",
        choices=model_defs.keys(),
        default=model_defs.keys(),
        help="Specify one or more models by name to evaluate.",
    )
    parser.add_argument(
        "--pixels",
        nargs="+",
        default=(1, 3, 5),
        type=int,
        help="The number of pixels that can be perturbed.",
    )
    parser.add_argument(
        "--maxiter",
        default=75,
        type=int,
        help="The maximum number of iterations in the differential evolution algorithm before giving up and failing the attack.",
    )
    parser.add_argument(
        "--popsize",
        default=400,
        type=int,
        help="The number of adversarial images generated each iteration in the differential evolution algorithm. Increasing this number requires more computation.",
    )
    parser.add_argument(
        "--samples",
        default=500,
        type=int,
        help="The number of image samples to attack. Images are sampled randomly from the dataset.",
    )
    parser.add_argument(
        "--targeted",
        action="store_true",
        help="Set this switch to test for targeted attacks.",
    )
    parser.add_argument(
        "--save",
        default="networks/results/results.pkl",
        help="Save location for the results (pickle)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print out additional information every iteration.",
    )
    parser.add_argument(
        "--defense",
        choices=["blur", "noise", "simclr", "all"],
        default=None,
        help="Choose the type of defense",
    )
    parser.add_argument(
        "--upload-results",
        action="store_true",
        help="Upload the results to huggingface_hub.",
    )
    parser.add_argument(
        "--minimize-change",
        action="store_true",
        help="Minimize the change in the image.",
    )

    args = parser.parse_args()

    # Load data and model
    _, test = cifar10.load_data()
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    models = [model_defs[m](load_weights=True) for m in args.model]

    def apply_selected_defense(image, defense_type):
        if defense_type == "blur":
            return apply_gaussian_blur(image)
        elif defense_type == "noise":
            return add_gaussian_noise(image)
        elif defense_type == "simclr":
            return simclr_augmentation(image)
        elif defense_type == "all":
            return simclr_augmentation(add_gaussian_noise(apply_gaussian_blur(image)))
        return image

    def predict_classes(xs, img, target_class, model, minimize=True):
        """
        This is the function to be minimized.
        RQ3 - TODO This function should be adjusted to minimize the change in the image

        xs: np.array (population_size, n_pixels * 5)
        img: np.array (32, 32, 3)
        target_clss: int
        model: keras.model
        minimize: bool
        """
        if args.defense:
            img = apply_selected_defense(img, args.defense)

        imgs_perturbed = helper.perturb_image(xs, img)
        predictions = model.predict(imgs_perturbed)[:, target_class]
        diff = imgs_perturbed - img
        predictions = predictions if minimize else 1 - predictions

        if args.minimize_change:
            change = np.mean(np.abs(diff), axis=(1, 2, 3))
            change = change if minimize else 1 - change
            predictions = predictions + change

        return predictions

    attacker = PixelAttacker(models, test, class_names)

    print("Starting attack")

    if args.deterministic_comparison:
        images = [1, 2, 3, 4, 5]
        results = []
        for img_id in images:
            for model in models:
                results.append(
                    attacker.attack(
                        model=model,
                        fitness_function=predict_classes,
                        img_id=img_id,
                        pixels=args.pixels,
                        targeted=args.targeted,
                        maxiter=args.maxiter,
                        popsize=args.popsize,
                        verbose=args.verbose,
                    )
                )

    else:
        results = attacker.attack_all(
            models=models,
            fitness_function=predict_classes,
            samples=args.samples,
            pixels=args.pixels,
            targeted=args.targeted,
            maxiter=args.maxiter,
            popsize=args.popsize,
            verbose=args.verbose,
        )
    # results = pickle.load(open("./baseline_results.pkl", "rb"))

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
    results_table = pd.DataFrame(results, columns=columns)

    print(results_table[["model", "pixels", "image", "true", "predicted", "success"]])

    for index, result in results_table[results_table["success"]].iterrows():
        # Subtract the original image to get the perturbation
        difference = np.abs(result["attack_image"] - result["original_image"])
        print("Mean: ", difference.mean())
        print("Max: ", difference.max())

    print("Saving to", args.save)
    with open(args.save, "wb") as file:
        pickle.dump(results, file)

    # ======================= RQ4 Analysis: One-Pixel Attack Consistency Across Groups =======================

    # Group results by model and class (true label) to analyze success rate
    grouped_results = results_table.groupby(["model", "true"])
    # Calculate success rate for each combination of model and class
    success_rate_by_group = grouped_results["success"].mean().reset_index()
    # Print the success rates for each group
    print("\nSuccess rate by model and class:")
    print(success_rate_by_group)

    # Further analysis: Check if specific classes are more vulnerable across all models
    class_success_rate = results_table.groupby("true")["success"].mean().reset_index()
    # Print the success rate per class (across all models)
    print("\nSuccess rate by class (across all models):")
    print(class_success_rate)

    # ======================= Bias Investigation: Misclassified Images by Class =======================
    misclassified_by_class = results_table[results_table["success"] == False].groupby("true").size()
    # Print out the misclassified count per class
    print("\nMisclassified images by class:")
    print(misclassified_by_class)

    # Analyze the impact of different pixel modifications across classes
    pixel_success_rate = results_table.groupby("pixels")["success"].mean().reset_index()
    # Print the success rate for different pixel perturbations
    print("\nSuccess rate by number of pixels perturbed:")
    print(pixel_success_rate)

    import seaborn as sns
    import matplotlib.pyplot as plt

    # Plot success rate by model and class (Bar Plot)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=success_rate_by_group, x="model", y="success", hue="true")
    plt.title("Success Rate of One-Pixel Attacks by Model and Class")
    plt.ylabel("Success Rate")
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    plt.legend(title="True Label", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

    # Plot success rate by class (Bar Plot)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=class_success_rate, x="true", y="success")
    plt.title("Success Rate of One-Pixel Attacks by Class")
    plt.ylabel("Success Rate")
    plt.xlabel("Class")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot success rate by number of pixels perturbed (Bar Plot)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=pixel_success_rate, x="pixels", y="success")
    plt.title("Success Rate of One-Pixel Attacks by Number of Pixels Perturbed")
    plt.ylabel("Success Rate")
    plt.xlabel("Number of Pixels Perturbed")
    plt.tight_layout()
    plt.show()

    # Save the success rate by group and per class to CSV for future reference
    success_rate_by_group.to_csv("success_rate_by_group.csv", index=False)
    class_success_rate.to_csv("class_success_rate.csv", index=False)

    from scipy import stats

    # --- Chi-Squared Test for Independence ---
    # Create a contingency table for 'model' vs 'success'
    contingency_table = pd.crosstab(results_table["model"], results_table["success"])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    print("\nChi-Squared Test for independence between model and attack success:")
    print(f"Chi2 statistic: {chi2}, p-value: {p_value}, degrees of freedom: {dof}")

    # --- One-Way ANOVA across classes ---
    groups = [group["success"].values for name, group in results_table.groupby("true")]
    f_stat, anova_p = stats.f_oneway(*groups)
    print("\nOne-Way ANOVA across classes for attack success rates:")
    print(f"F-statistic: {f_stat}, p-value: {anova_p}")

    # --- Heatmap Visualization of Success Rate by Model and Class ---
    heatmap_data = success_rate_by_group.pivot(index="model", columns="true", values="success")
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="viridis", cbar=True)
    plt.title("Heatmap: Success Rate of One-Pixel Attacks by Model and Class")
    plt.xlabel("True Class")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.show()

    # Function to visualize original and perturbed images side by side
    def visualize_image_comparison(original_image, perturbed_image, true_label):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(original_image)
        ax[0].set_title(f"Original: {class_names[true_label]}")
        ax[0].axis("off")
        ax[1].imshow(perturbed_image)
        ax[1].set_title(f"Perturbed: {class_names[true_label]}")
        ax[1].axis("off")
        plt.tight_layout()
        plt.show()

    # Display side-by-side comparisons for a few successful attacks
    sample_successes = results_table[results_table["success"] == True].sample(n=5, random_state=42)
    for index, row in sample_successes.iterrows():
        visualize_image_comparison(row["original_image"], row["attack_image"], row["true"])
