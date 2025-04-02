#!/usr/bin/env python3

import argparse

import numpy as np
import random
import pandas as pd
from tensorflow.keras.datasets import cifar10
import pickle

# Custom Networks
from networks.lenet import LeNet
from networks.pure_cnn import PureCnn
from networks.network_in_network import NetworkInNetwork
from networks.resnet import ResNet
from networks.densenet import DenseNet
from networks.wide_resnet import WideResNet
from networks.capsnet import CapsNet

# Helper functions
from differential_evolution import differential_evolution
import helper


class PixelAttacker:
    def __init__(self, models, data, class_names, dimensions=(32, 32)):
        # Load data and model
        self.models = models
        self.x_test, self.y_test = data
        self.class_names = class_names
        self.dimensions = dimensions

        network_stats, correct_imgs = helper.evaluate_models(
            self.models, self.x_test, self.y_test
        )
        self.correct_imgs = pd.DataFrame(
            correct_imgs, columns=pd.Index(["name", "img", "label", "confidence", "pred"])
        )
        self.network_stats = pd.DataFrame(
            network_stats, columns=pd.Index(["name", "accuracy", "param_count"])
        )

    def attack_success(
        self, x, img, target_class, model, targeted_attack=False, verbose=False
    ):
        # Perturb the image with the given pixel(s) and get the prediction of the model
        attack_image = helper.perturb_image(x, img)

        confidence = model.predict(attack_image)[0]
        predicted_class = np.argmax(confidence)

        # If the prediction is what we want (misclassification or
        # targeted classification), return True
        if verbose:
            print("Confidence:", confidence[target_class])
        if (targeted_attack and predicted_class == target_class) or (
            not targeted_attack and predicted_class != target_class
        ):
            return True

    def attack(
        self,
        fitness_function,
        img_id,
        model,
        target=None,
        pixel_count=1,
        maxiter=75,
        popsize=400,
        verbose=False,
        plot=False,
    ):
        # Change the target class based on whether this is a targeted attack or not
        targeted_attack = target is not None
        target_class = target if targeted_attack else self.y_test[img_id, 0]

        # Define bounds for a flat vector of x,y,r,g,b values
        # For more pixels, repeat this layout
        dim_x, dim_y = self.dimensions
        bounds = [(0, dim_x), (0, dim_y), (0, 256), (0, 256), (0, 256)] * pixel_count

        # Population multiplier, in terms of the size of the perturbation vector x
        popmul = max(1, popsize // len(bounds))

        # Format the predict/callback functions for the differential evolution algorithm
        def predict_fn(xs):
            # TODO adjust this for RQ1
            return fitness_function(
                xs, self.x_test[img_id], target_class, model, target is None
            )

        def callback_fn(x, convergence):
            return self.attack_success(
                x, self.x_test[img_id], target_class, model, targeted_attack, verbose
            )

        # Call Scipy's Implementation of Differential Evolution
        with helper.suppress_stdout(verbose=verbose):
            attack_result = differential_evolution(
                predict_fn,
                bounds,
                maxiter=maxiter,
                popsize=popmul,
                recombination=1,
                atol=-1,
                callback=callback_fn,
                polish=False,
            )

        # Calculate some useful statistics to return from this function
        original_img = self.x_test[img_id]
        attack_image = helper.perturb_image(attack_result.x, original_img)[0]
        prior_probs = model.predict(np.array([self.x_test[img_id]]))[0]
        predicted_probs = model.predict(np.array([attack_image]))[0]
        predicted_class = np.argmax(predicted_probs)
        actual_class = self.y_test[img_id, 0]
        success = predicted_class != actual_class
        cdiff = prior_probs[actual_class] - predicted_probs[actual_class]

        # Show the best attempt at a solution (successful or not)
        if plot:
            helper.plot_image(
                attack_image, actual_class, self.class_names, predicted_class
            )

        return [
            model.name,
            attack_image,
            original_img,
            pixel_count,
            img_id,
            actual_class,
            predicted_class,
            success,
            cdiff,
            prior_probs,
            predicted_probs,
            attack_result.x,
        ]

    def attack_all(
        self,
        models,
        fitness_function,
        samples=500,
        pixels=(1, 3, 5),
        targeted=False,
        maxiter=75,
        popsize=400,
        verbose=False,
        selected_imgs=None,
    ):
        results = []
        for model in models:
            model_results = []
            valid_imgs = self.correct_imgs[self.correct_imgs.name == model.name].img
            if selected_imgs:
                img_samples = np.random.choice(valid_imgs, samples)
            else:
                img_samples = selected_imgs

            for pixel_count in pixels:
                for i, img in enumerate(img_samples):
                    print(model.name, "- image", img, "-", i + 1, "/", len(img_samples))
                    targets = [None] if not targeted else range(10)

                    for target in targets:
                        if targeted:
                            print("Attacking with target", self.class_names[target])
                            if target == self.y_test[img, 0]:
                                continue
                        result = self.attack(
                            fitness_function,
                            img,
                            model,
                            target,
                            pixel_count,
                            maxiter=maxiter,
                            popsize=popsize,
                            verbose=verbose,
                        )
                        model_results.append(result)

            results += model_results
            helper.checkpoint(results, targeted)
        return results


