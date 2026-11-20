"""
FreshSense Model Training Script (v2 - stronger augmentation + fine-tuning)
=============================================================================
This version is built to generalize better to real-world photos (e.g. ones
downloaded from Google Images, taken on a phone, etc.) instead of only
performing well on test images from the same dataset it trained on.

Key improvements over v1:
  - Heavier data augmentation (rotation, zoom, contrast, brightness) baked
    directly into the model, not just left-right flipping.
  - Two-phase training: frozen backbone first, then fine-tune the TOP
    portion of the MobileNetV2 backbone (not the whole thing) at a low
    learning rate, so it actually learns spoilage-specific visual cues
    instead of relying only on generic ImageNet features.
  - Class weighting, so rarer classes (e.g. a fruit with fewer "rotten"
    photos than others) don't get ignored during training.
  - Early stopping + learning-rate reduction, so training doesn't overfit
    to the training set the longer it runs.
  - Saves metadata.json + metrics.json in the same format the FreshSense
    backend already expects (so nothing else in the project needs to change).

RECOMMENDED DATASETS (search these on Kaggle):
  - "Fruits fresh and rotten for classification" (Sriram Reddy)
  - "Fruit and Vegetable Disease (Healthy vs Rotten)"

IMPORTANT - the single biggest thing that actually fixes real-world
generalization: mix in extra "in the wild" photos (phone photos, Google
Images results) into your train/ folders yourself, not just relying on
augmentation. Augmentation makes existing photos more varied; it can't
invent spoilage patterns the model has never seen. Aim for at least a
few dozen extra varied photos per class if real-world accuracy matters
more than a quick demo.

EXPECTED FOLDER STRUCTURE (ImageFolder style):

dataset/
  train/
    apple_fresh/
    apple_rotten/
    banana_fresh/
    banana_rotten/
    ... etc for all fruits
  val/
    apple_fresh/
    apple_rotten/
    ... etc (same folder names as train/)

Naming convention matters: "<food>_<condition>" where condition is
"fresh" or "rotten" (or "moderate" if you have a 3-way label per fruit).

OPTIONAL BUT RECOMMENDED - teach the model what is NOT a fruit:
add one extra folder named exactly   not_fruit   to BOTH train/ and val/ and
fill it with a few hundred varied photos of other things (people, cars, bottles,
books, tables, vegetables, gravel, brick walls, kitchen scenes ...). The backend
recognises this class automatically and answers "this is not a fruit" for such
photos - no code change needed. Without it, the app relies on a score threshold,
which can still be fooled by texture-like images (e.g. gravel looks like grapes).

USAGE:
    python train.py --data_dir ./dataset --epochs 15 --fine_tune_epochs 10 --output ../backend/model

This produces, inside --output:
    fruit_condition.keras   (the trained model)
    metadata.json            (class list + input settings, read by the backend)
    metrics.json              (accuracy breakdown, for your own reference)
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


def build_model(num_classes, img_size=224):
    """
    Builds a model with augmentation + preprocessing baked in, so the
    backend can feed it raw 0..255 RGB images directly (no normalization
    needed outside the model).
    """
    inputs = layers.Input(shape=(img_size, img_size, 3))

    # --- Augmentation (training-time only; inert at inference) ---
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.15)(x)
    x = layers.RandomZoom(0.2)(x)
    x = layers.RandomContrast(0.2)(x)
    try:
        x = layers.RandomBrightness(0.2)(x)
    except AttributeError:
        # Older TF/Keras versions may not have RandomBrightness; skip gracefully.
        pass

    # --- Preprocessing: maps 0..255 -> -1..1, matching MobileNetV2's expectations ---
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)

    base = MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet")
    base.trainable = False  # phase 1: frozen

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


def compute_class_weights(train_dir, class_names):
    """Weight rarer classes higher so the model doesn't ignore them."""
    counts = []
    for name in class_names:
        folder = os.path.join(train_dir, name)
        n = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
        counts.append(max(n, 1))
    total = sum(counts)
    n_classes = len(counts)
    weights = {i: total / (n_classes * c) for i, c in enumerate(counts)}
    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to dataset root (with train/ and val/ subfolders)")
    parser.add_argument("--epochs", type=int, default=15, help="Phase 1 epochs (frozen backbone)")
    parser.add_argument("--fine_tune_epochs", type=int, default=10, help="Phase 2 epochs (fine-tuning)")
    parser.add_argument("--unfreeze_layers", type=int, default=40,
                         help="How many of the TOP backbone layers to unfreeze during fine-tuning")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--output", default="../backend/model", help="Where to save the trained model + metadata")
    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=(args.img_size, args.img_size), batch_size=args.batch_size, label_mode="categorical"
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=(args.img_size, args.img_size), batch_size=args.batch_size, label_mode="categorical"
    )

    class_names = train_ds_raw.class_names
    print(f"Detected classes ({len(class_names)}): {class_names}")

    class_weights = compute_class_weights(train_dir, class_names)
    print(f"Class weights (higher = rarer class, weighted more): {class_weights}")

    # NOTE: images are intentionally left as raw 0..255 here -- the model itself
    # contains the Rescaling layer, so the backend can feed raw pixels straight in.
    train_ds = train_ds_raw.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    model, base = build_model(num_classes=len(class_names), img_size=args.img_size)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ]

    print("\n--- Phase 1: training classifier head (backbone frozen) ---")
    model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs,
        class_weight=class_weights, callbacks=callbacks,
    )

    print(f"\n--- Phase 2: fine-tuning top {args.unfreeze_layers} backbone layers ---")
    base.trainable = True
    for layer in base.layers[:-args.unfreeze_layers]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs,
        class_weight=class_weights, callbacks=callbacks,
    )

    # --- Evaluate + build metrics.json (fruit / condition broken out separately) ---
    print("\n--- Evaluating on validation set ---")
    y_true, y_pred = [], []
    for batch_x, batch_y in val_ds:
        preds = model.predict(batch_x, verbose=0)
        y_true.extend(np.argmax(batch_y.numpy(), axis=1).tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    def parse(idx):
        name = class_names[idx]
        fruit, condition = name.rsplit("_", 1)
        return fruit, condition

    fruit_correct = sum(1 for t, p in zip(y_true, y_pred) if parse(t)[0] == parse(p)[0])
    condition_correct = sum(1 for t, p in zip(y_true, y_pred) if parse(t)[1] == parse(p)[1])
    overall_correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    n = len(y_true)

    metrics = {
        "test_accuracy": overall_correct / n if n else 0,
        "fruit_accuracy": fruit_correct / n if n else 0,
        "condition_accuracy": condition_correct / n if n else 0,
        "classes": [{"fruit": f, "condition": c} for f, c in (parse(i) for i in range(len(class_names)))],
    }

    # --- Save everything in the format the backend expects ---
    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, "fruit_condition.keras")
    metadata_path = os.path.join(args.output, "metadata.json")
    metrics_path = os.path.join(args.output, "metrics.json")

    model.save(model_path)

    metadata = {
        "classes": [{"fruit": f, "condition": c} for f, c in (parse(i) for i in range(len(class_names)))],
        "image_size": [args.img_size, args.img_size],
        "task": "fruit_and_condition",
        "model": "MobileNetV2, fine-tuned top layers + augmentation",
        "input": "RGB pixels 0..255",
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model to {model_path}")
    print(f"Saved metadata to {metadata_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"\nOverall accuracy: {metrics['test_accuracy']:.3f}  "
          f"(fruit ID: {metrics['fruit_accuracy']:.3f}, fresh/rotten: {metrics['condition_accuracy']:.3f})")
    print("\nRestart the Flask backend to pick up the new model automatically.")


if __name__ == "__main__":
    main()
