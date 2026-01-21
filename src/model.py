import numpy as np
import joblib
import os
from keras.datasets import mnist
from sklearn.svm import LinearSVC

def train_and_save_model():
    print("Loading MNIST dataset...")

    (X_train, y_train), _ = mnist.load_data()

    # Use SMALL subset for fast training
    X_train = X_train[:5000]
    y_train = y_train[:5000]

    # Flatten 28x28 -> 784
    X_train = X_train.reshape(-1, 784)

    # Normalize
    X_train = X_train / 255.0

    print("Training fast Linear SVM model...")

    model = LinearSVC(max_iter=5000)
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/svm_model.pkl")

    print("✅ Model trained and saved successfully")
