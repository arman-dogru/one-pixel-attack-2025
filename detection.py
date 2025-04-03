import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, Add, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Updated Data Loading Function with Grouping
def load_detection_dataset(result_path="baseline_results.pkl", random_state=42):
    df = pd.read_pickle(result_path)
    # Filter to include only rows where the attack was successful,
    # so each row defines a valid pair (clean and adversarial)
    df = df[df["success"] == True].reset_index(drop=True)
    adv_imgs = np.stack(df["attack_image"].values)
    clean_imgs = np.stack(df["original_image"].values)
    
    # Concatenate clean and adversarial images into one array
    X = np.concatenate([clean_imgs, adv_imgs], axis=0)
    y = np.array([0] * len(clean_imgs) + [1] * len(adv_imgs))
    X = X.astype('float32') / 255.0
    
    # Create group IDs so that the clean and adversarial version share the same group.
    # Since each row is one pair, we create one group per row and repeat it twice.
    groups = np.repeat(np.arange(len(df)), 2)
    return X, y, groups

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

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.show()

# Neural Network Model Definitions
def build_simple_cnn(input_shape=(32, 32, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape),
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

def build_deeper_cnn(input_shape=(32, 32, 3)):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten(name="flatten")(x)  # Named layer for feature extraction
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

def build_resnet_like(input_shape=(32, 32, 3)):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    residual = Conv2D(64, (1, 1), strides=(2, 2), padding="same")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Add()([x, residual])
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

def build_mlp(input_shape=(32, 32, 3)):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

def build_vgg16_finetune(input_shape=(224, 224, 3)):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base model layers
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

# Traditional ML Model Definitions
def build_svm():
    return SVC(kernel='rbf', probability=True)

def build_random_forest():
    return RandomForestClassifier(n_estimators=100)

def build_logistic_regression():
    return LogisticRegression(max_iter=1000)

def show_sample_images(X, y, num_samples=5):
    clean_indices = np.where(y == 0)[0][:num_samples]
    adv_indices = np.where(y == 1)[0][:num_samples]
    
    plt.figure(figsize=(2 * num_samples, 4))
    for i, idx in enumerate(clean_indices):
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(X[idx])
        plt.axis("off")
        plt.title("Clean")
    
    for i, idx in enumerate(adv_indices):
        plt.subplot(2, num_samples, num_samples + i + 1)
        plt.imshow(X[idx])
        plt.axis("off")
        plt.title("Adversarial")
    
    plt.suptitle("Clean vs. One-Pixel Attacked Images", fontsize=16)
    plt.tight_layout()
    plt.show()

# Main Training and Evaluation Function with Group-Based Splitting
def train_and_evaluate_multiple_models(result_path="baseline_results.pkl", epochs=20, batch_size=32):
    # Load and explore dataset with group IDs
    X, y, groups = load_detection_dataset(result_path)
    explore_detection_data(X, y)
    
    # Group-based splitting: 70% training, 20% validation, 10% test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(gss.split(X, y, groups))
    X_train, y_train, groups_train = X[train_idx], y[train_idx], groups[train_idx]
    X_temp, y_temp, groups_temp = X[temp_idx], y[temp_idx], groups[temp_idx]
    
    gss2 = GroupShuffleSplit(n_splits=1, test_size=1/3, random_state=42)
    val_idx, test_idx = next(gss2.split(X_temp, y_temp, groups_temp))
    X_val, y_val = X_temp[val_idx], y_temp[val_idx]
    X_test, y_test = X_temp[test_idx], y_temp[test_idx]
    
    # Visualize a few samples
    show_sample_images(X, y)

    # Define neural network models
    nn_models = {
        "Simple CNN": build_simple_cnn,
        "Deeper CNN": build_deeper_cnn,
        "ResNet-like": build_resnet_like,
        "MLP": build_mlp,
        "VGG16": build_vgg16_finetune
    }
    
    # Define traditional ML models
    ml_models = {
        "SVM": build_svm,
        "Random Forest": build_random_forest,
        "Logistic Regression": build_logistic_regression
    }
    
    # Train Deeper CNN for feature extraction using the validation set
    print("\nTraining Deeper CNN for feature extraction...")
    deeper_cnn = build_deeper_cnn(input_shape=X.shape[1:])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    deeper_cnn.fit(X_train, y_train, validation_data=(X_val, y_val),
                   epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[early_stopping])
    feature_extractor = Model(inputs=deeper_cnn.input, outputs=deeper_cnn.get_layer('flatten').output)
    X_train_features = feature_extractor.predict(X_train)
    X_test_features = feature_extractor.predict(X_test)
    
    # Train and evaluate neural network models
    for name, build_model in nn_models.items():
        print(f"\nTraining {name}...")
        if name == "VGG16":
            # Resize images for VGG16
            X_train_resized = tf.image.resize(X_train, (224, 224))
            X_val_resized = tf.image.resize(X_val, (224, 224))
            X_test_resized = tf.image.resize(X_test, (224, 224))
            model = build_model(input_shape=(224, 224, 3))
            history = model.fit(X_train_resized, y_train, validation_data=(X_val_resized, y_val),
                                epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[early_stopping])
            y_pred_prob = model.predict(X_test_resized)
        else:
            model = build_model(input_shape=X.shape[1:])
            history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[early_stopping])
            y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"{name} Performance on Test Set:")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
        if name != "VGG16":  # VGG16 uses a different input size, so skip plotting for it
            plot_training_history(history)
    
    # Train and evaluate traditional ML models
    for name, ml_model in ml_models.items():
        print(f"\nTraining {name}...")
        model = ml_model()
        model.fit(X_train_features, y_train)
        if hasattr(model, "predict_proba"):
            y_pred_prob = model.predict_proba(X_test_features)[:, 1]
        else:
            y_pred_prob = model.decision_function(X_test_features)
        y_pred = (y_pred_prob > 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"{name} Performance on Test Set:")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")

if __name__ == "__main__":
    train_and_evaluate_multiple_models()
