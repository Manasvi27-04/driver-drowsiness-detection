"""
train.py

Two-phase training:
  Phase 1: Train classification head with frozen MobileNetV2 base.
  Phase 2: Unfreeze top layers and fine-tune with a low learning rate.
"""

import argparse
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from data_augmentation import get_train_generator, get_eval_generator
from model import build_model, compile_model, unfreeze_top_layers


def main(data_dir, epochs, batch_size, model_out):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_gen = get_train_generator(train_dir, batch_size)
    val_gen = get_eval_generator(val_dir, batch_size)

    model, base_model = build_model(freeze_base=True)
    model = compile_model(model, learning_rate=1e-3)

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    callbacks = [
        ModelCheckpoint(model_out, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]

    print("=== Phase 1: Training classification head (frozen base) ===")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=max(1, epochs // 2),
        callbacks=callbacks,
    )

    print("=== Phase 2: Fine-tuning top layers of MobileNetV2 ===")
    unfreeze_top_layers(base_model, num_layers=30)
    model = compile_model(model, learning_rate=1e-5)
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        initial_epoch=max(1, epochs // 2),
        callbacks=callbacks,
    )

    model.save(model_out)
    print(f"Model saved to {model_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train driver drowsiness detection model")
    parser.add_argument("--data_dir", required=True, help="Directory with train/ and val/ subfolders")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_out", default="models/drowsiness_mobilenetv2.h5")
    args = parser.parse_args()
    main(args.data_dir, args.epochs, args.batch_size, args.model_out)
