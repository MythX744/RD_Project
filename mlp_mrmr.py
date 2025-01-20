import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from skimage.io import imread
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
import io
from datetime import datetime
import random


def load_all_images(image_dir, label_dir, selected_bands=None):
    """
    Load random image and label pairs from the directories and select specific bands.

    Parameters:
    - image_dir (str): Directory containing the image files.
    - label_dir (str): Directory containing the label files.
    - n (int): Number of image-label pairs to load.
    - selected_bands (list or None): List of band indices to select. If None, keep all bands.

    Returns:
    - all_images (np.ndarray): Array of selected bands from the images.
    - all_labels (np.ndarray): Array of labels.
    """
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])
    all_images = []
    all_labels = []

    print(f"Found {len(image_files)} image files")

    for img_file in image_files:
        print(f"Processing {img_file}")
        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, img_file)

        if os.path.exists(label_path):
            img_array = imread(img_path)
            label_array = imread(label_path)

            # Select specific bands if provided
            if selected_bands is not None:
                img_array = img_array[:, :, selected_bands]

            all_images.append(img_array)
            all_labels.append(label_array)

    return np.array(all_images), np.array(all_labels)


def main():
    tf.keras.backend.clear_session()

    # Set paths
    image_dir = 'C:/Users/malak/Documents/CV_project/Code/malak_images'
    label_dir = 'C:/Users/malak/Documents/CV_project/Code/malak_landcover'

    selected_bands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Create output directory for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'deep_mlp_all_images_full_score_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load all images and labels
        print("Loading images and labels...")
        images, labels = load_all_images(image_dir, label_dir, selected_bands=selected_bands)
        print(f"Loaded {len(images)} image-label pairs")
        print(f"Image shape: {images.shape}")
        print(f"Label shape: {labels.shape}")

        # Prepare data
        print("\nPreparing data...")
        n_samples = images.shape[0] * images.shape[1] * images.shape[2]
        n_features = images.shape[3]
        pixels = images.reshape(-1, n_features)

        # Normalize and scale the data
        pixels = pixels / 255.0
        scaler = StandardScaler()
        pixels = scaler.fit_transform(pixels)

        # Flatten labels
        flat_labels = labels.reshape(-1)
        print(f"Prepared data shape - Features: {pixels.shape}, Labels: {flat_labels.shape}")

        # Create class mapping
        unique_classes = np.unique(flat_labels)
        class_mapping = {label: idx for idx, label in enumerate(unique_classes)}
        n_classes = len(class_mapping)
        print(f"\nFound {n_classes} unique classes")

        # Save class mapping
        with open(os.path.join(output_dir, 'class_mapping.txt'), 'w', encoding='utf-8') as f:
            for original, mapped in class_mapping.items():
                f.write(f"Original: {original} -> Mapped: {mapped}\n")

        print("\nShuffling data...")
        shuffled_indices = np.random.permutation(len(pixels))
        pixels = pixels[shuffled_indices]
        flat_labels = flat_labels[shuffled_indices]

        # Split data
        pixel_train, pixel_test, label_train, label_test = train_test_split(
            pixels, flat_labels, test_size=0.2, random_state=42
        )

        # Map labels and convert to one-hot encoding
        label_train = np.array([class_mapping[label] for label in label_train])
        label_test = np.array([class_mapping[label] for label in label_test])
        label_train_onehot = tf.keras.utils.to_categorical(label_train, num_classes=n_classes)
        label_test_onehot = tf.keras.utils.to_categorical(label_test, num_classes=n_classes)

        # Define and compile model
        print("\nCreating model...")
        model = tf.keras.models.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=(n_features,)),

            # First hidden layer
            tf.keras.layers.Dense(256),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.25),

            # Second hidden layer
            tf.keras.layers.Dense(256),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.25),

            # Third hidden layer
            tf.keras.layers.Dense(128),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.25),

            # Fourth hidden layer
            tf.keras.layers.Dense(64),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dropout(0.25),

            # Output layer
            tf.keras.layers.Dense(n_classes, activation='softmax')
        ])

        # Learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Save model summary
        with open(os.path.join(output_dir, 'model_summary.txt'), 'w', encoding='utf-8') as f:
            string_buffer = io.StringIO()
            model.summary(print_fn=lambda x: string_buffer.write(x + '\n'))
            f.write(string_buffer.getvalue())

        model.summary()

        # Train model
        history = model.fit(
            pixel_train,
            label_train_onehot,
            epochs=50,
            batch_size=512,
            validation_split=0.2,
            verbose=1
        )

        # Save training history plot
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'training_history.png'))
        plt.close()

        # Evaluate model
        print("\nEvaluating model...")
        test_loss, test_accuracy = model.evaluate(pixel_test, label_test_onehot, verbose=1)

        # Save metrics
        with open(os.path.join(output_dir, 'test_metrics.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Test Loss: {test_loss}\n")
            f.write(f"Test Accuracy: {test_accuracy}\n")

        # Generate predictions and classification report
        predictions = []
        batch_size = 512
        for i in range(0, len(pixel_test), batch_size):
            batch = pixel_test[i:i + batch_size]
            batch_pred = model.predict(batch, verbose=0)
            predictions.append(batch_pred)

        test_predictions = np.vstack(predictions)
        predicted_labels = np.argmax(test_predictions, axis=1)

        # Save classification report
        report = classification_report(label_test, predicted_labels)
        print("\nClassification Report:")
        print(report)

        with open(os.path.join(output_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\nTest Accuracy: {test_accuracy}")

        # Save model
        model.save(os.path.join(output_dir, 'mlp_model.h5'))
        print(f"\nResults saved in directory: {output_dir}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
