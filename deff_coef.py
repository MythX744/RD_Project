import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from skimage.io import imread
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import io


def load_all_images(image_dir, label_dir):
    """Load all image and label pairs from the directories"""
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

            all_images.append(img_array)
            all_labels.append(label_array)

    return np.array(all_images), np.array(all_labels)


def select_bands(images):
    """Select bands using the specified formula in a memory-efficient way"""
    # Reshape the images to 2D array (pixels x bands)
    n_samples = images.shape[0] * images.shape[1] * images.shape[2]
    n_features = images.shape[3]
    pixels = images.reshape(-1, n_features)

    # Process each band separately to avoid memory issues
    band_deflections = []

    for band_idx in range(n_features):
        band_data = pixels[:, band_idx]
        # Calculate a single deflection value per band
        # Using mean as a simple metric - replace with your deflection calculation
        deflection = np.mean(band_data)
        band_deflections.append({
            'band': band_idx,
            'deflection': deflection
        })

    # Create DataFrame with just one row per band
    df = pd.DataFrame(band_deflections)

    # Apply your selection formula
    selected_bands = (df.sort_values('deflection', ascending=True)
                        .groupby('band')
                        .head(1)
                        .sort_values('deflection', ascending=False)
                        .head(10)['band'])

    return selected_bands.tolist()


def main():
    tf.keras.backend.clear_session()

    # Set paths
    image_dir = 'C:/Users/malak/Documents/CV_project/Code/malak_images'
    label_dir = 'C:/Users/malak/Documents/CV_project/Code/malak_landcover'

    # Create output directory for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'def_coef_results_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load images and labels
        print("Loading images and labels...")
        images, labels = load_all_images(image_dir, label_dir)
        print(f"Loaded {len(images)} image-label pairs")
        print(f"Image shape: {images.shape}")
        print(f"Label shape: {labels.shape}")

        # Perform band selection
        print("\nPerforming band selection...")
        selected_bands = select_bands(images)
        print(f"Selected bands: {selected_bands}")

        # Use only selected bands
        images = images[:, :, :, selected_bands]
        print(f"New image shape after band selection: {images.shape}")

        # Prepare data
        print("\nPreparing data...")
        n_samples = images.shape[0] * images.shape[1] * images.shape[2]
        n_features = len(selected_bands)  # Now using only selected bands
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

        # Save band selection results
        with open(os.path.join(output_dir, 'selected_bands.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Selected bands: {selected_bands}\n")
            f.write(f"Number of bands: {len(selected_bands)}\n")

        print("\nShuffling data...")
        shuffled_indices = np.random.permutation(len(pixels))
        pixels = pixels[shuffled_indices]
        flat_labels = flat_labels[shuffled_indices]

        # Split data
        pixel_train, pixel_test, label_train, label_test = train_test_split(
            pixels, flat_labels, test_size=0.2, random_state=42, stratify=flat_labels
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
