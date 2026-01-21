import os
from src.model import train_and_save_model
from src.gui import launch_gui

MODEL_PATH = "models/svm_model.pkl"

if not os.path.exists(MODEL_PATH):
    train_and_save_model()

launch_gui()
