"""
Detection of One-Pixel Attacks (RQ2)

This module implements a secondary model to detect whether an input image 
has been perturbed by a one-pixel attack, targeting 99% accuracy without overfitting.
"""

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def load_detection_dataset(result_path="baseline_results.pkl", random_state=42):
    # Load results into a DataFrame
    df = pd.read_pickle(result_path)
    adv_df = df[df["success"] == True]
    adv_imgs = np.stack(adv_df["attack_image"].values)
    clean_imgs = np.stack(df["original_image"].values)
    if clean_imgs.shape[0] > adv_imgs.shape[0]:
        np.random.seed(random_state)
        idx = np.random.choice(clean_imgs.shape[0], adv_imgs.shape[0], replace=False)
        clean_imgs = clean_imgs[idx]
    X = np.concatenate([clean_imgs, adv_imgs], axis=0)
    y = np.array([0] * clean_imgs.shape[0] + [1] * adv_imgs.shape[0])
    X = X.astype('float32') / 255.0
    return X, y

def explore_detection_data(X, y):
    total_samples = X.shape[0]
    unique, counts = np.unique(y, return_counts=True)
    print(f"Total samples: {total_samples}")
    for label, count in zip(unique, counts):
        label_name = "Clean (0)" if label == 0 else "Adversarial (1)"
        print(f"{label_name}: {count} samples")
    plt.figure(figsize=(6, 4))
    plt.bar(unique, counts, tick_label=["Clean", "Adversarial"])
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Label Distribution")
    plt.show()

def build_detection_model(input_shape=(32, 32, 3)):
    """
    Build a CNN with regularization to prevent overfitting.
    - Added BatchNormalization for training stability.
    - Retained Dropout for regularization.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss to check for overfitting.
    """
    plt.figure(figsize=(12, 4))
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.show()

def train_and_evaluate_detector(result_path="baseline_results.pkl", test_size=0.2, epochs=20, batch_size=32):
    # Load and explore dataset
    X, y = load_detection_dataset(result_path)
    explore_detection_data(X, y)
    
    # Split dataset (stratified to maintain balance)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Build model
    model = build_detection_model(input_shape=X.shape[1:])
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train model with validation and early stopping
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=epochs, batch_size=batch_size, verbose=2,
                        callbacks=[early_stopping])
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("Detection Model Performance:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    return model, history

if __name__ == "__main__":
    trained_model, history = train_and_evaluate_detector()