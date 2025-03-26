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
        write_permission=True,  # Requires a token with write permission
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
        "--defence",
        action="store_true",
        help="Set this switch to test the defense against the attack.",
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

    def predict_classes(xs, img, target_class, model, minimize=True):
        """
        This is the function to be minimized.
        RQ3 - TODO This function should be adjusted to minimize the change in the image

        xs: np.array (population_size, n_pixels * 5)
        img: np.array (32, 32, 3)
        target_class: int
        model: keras.model
        minimize: bool
        """
        # Perturb the image with the given pixel(s) x and get the prediction of the model
        imgs_perturbed = helper.perturb_image(
            xs, img
        )  # Shape: (population_size, 32, 32, 3)
        predictions = model.predict(imgs_perturbed)[:, target_class]

        # Change pixel values of perturbed image from original image
        # L1
        diff = imgs_perturbed - img  # Shape: (population_size, 32, 32, 3)
        # L2
        # Mean of the change in pixel values
        change = np.mean(np.abs(diff), axis=(1, 2, 3))
        print(change)

        # This function should always be minimized, so return its complement if needed
        predictions = (
            predictions if minimize else 1 - predictions
        )  # Confidence should go down
        change = change if minimize else 1 - change  # Change should go down
        score = predictions + change
        return score

    attacker = PixelAttacker(models, test, class_names)

    print("Starting attack")

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
