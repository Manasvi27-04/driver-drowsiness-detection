"""
evaluate.py

Evaluates a trained model on the test set and prints classification
metrics + a confusion matrix.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

from data_augmentation import get_eval_generator

CLASS_NAMES = ["alert", "sleepy", "yawning"]


def main(model_path, test_dir, output_fig="confusion_matrix.png"):
    model = load_model(model_path)
    test_gen = get_eval_generator(test_dir, shuffle=False)

    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes

    print("=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Driver Drowsiness Detection")
    plt.tight_layout()
    plt.savefig(output_fig)
    print(f"Confusion matrix saved to {output_fig}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate driver drowsiness detection model")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--output_fig", default="confusion_matrix.png")
    args = parser.parse_args()
    main(args.model_path, args.test_dir, args.output_fig)
