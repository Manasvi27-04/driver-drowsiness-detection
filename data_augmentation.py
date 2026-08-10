"""
data_augmentation.py

Defines the Keras ImageDataGenerator pipelines used for training and
validation/test. Augmentation is applied only to the training set to
improve generalization across lighting conditions, head angles, and
camera positions.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def get_train_generator(train_dir: str, batch_size: int = BATCH_SIZE):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        brightness_range=(0.7, 1.3),
        horizontal_flip=True,
        fill_mode="nearest",
    )
    return train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )


def get_eval_generator(eval_dir: str, batch_size: int = BATCH_SIZE, shuffle: bool = False):
    """Used for validation and test sets — rescale only, no augmentation."""
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)
    return eval_datagen.flow_from_directory(
        eval_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=shuffle,
    )
