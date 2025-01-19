import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from skimage.io import imread
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import io
from datetime import datetime


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


def create_unet(input_shape, n_classes):
    """Create U-Net model"""

    def conv_block(inputs, filters):
        x = tf.keras.layers.Conv2D(filters, 3, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(0.1)(x)

        x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        return x

    # Input
    inputs = tf.keras.layers.Input(input_shape)

    # Encoder
    conv1 = conv_block(inputs, 64)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, 128)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, 256)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(pool3, 512)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bridge
    conv5 = conv_block(pool4, 1024)

    # Decoder
    up6 = tf.keras.layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(conv5)
    concat6 = tf.keras.layers.Concatenate()([conv4, up6])
    conv6 = conv_block(concat6, 512)

    up7 = tf.keras.layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    concat7 = tf.keras.layers.Concatenate()([conv3, up7])
    conv7 = conv_block(concat7, 256)

    up8 = tf.keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    concat8 = tf.keras.layers.Concatenate()([conv2, up8])
    conv8 = conv_block(concat8, 128)

    up9 = tf.keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    concat9 = tf.keras.layers.Concatenate()([conv1, up9])
    conv9 = conv_block(concat9, 64)

    # Output
    outputs = tf.keras.layers.Conv2D(n_classes, 1, activation='softmax')(conv9)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


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
            img_array = img_array.astype(np.float32)
            img_array = (img_array - np.mean(img_array)) / (np.std(img_array) + 1e-8)

            label_array = imread(label_path)

            all_images.append(img_array)
            all_labels.append(label_array)

    return np.array(all_images), np.array(all_labels)


def main():
    tf.keras.backend.clear_session()

    # Set paths
    image_dir = 'C:/Users/malak/Documents/CV_project/Code/malak_images'
    label_dir = 'C:/Users/malak/Documents/CV_project/Code/malak_landcover'

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'unet_defcoef_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load all images first
        print("Loading images and labels...")
        images, labels = load_all_images(image_dir, label_dir)
        print(f"Loaded {len(images)} image-label pairs")
        print(f"Original image shape: {images.shape}")

        # Select bands
        print("\nSelecting bands...")
        selected_bands = select_bands(images)
        print(f"Selected bands: {selected_bands}")

        # Save selected bands information
        with open(os.path.join(output_dir, 'selected_bands.txt'), 'w', encoding='utf-8') as f:
            f.write("Selected bands:\n")
            for band in selected_bands:
                f.write(f"Band {band}\n")

        # Use only selected bands
        images = images[:, :, :, selected_bands]
        print(f"Image shape after band selection: {images.shape}")

        # Create class mapping
        unique_classes = np.unique(labels)
        class_mapping = {label: idx for idx, label in enumerate(unique_classes)}
        n_classes = len(class_mapping)
        print(f"\nFound {n_classes} unique classes")

        # Save class mapping
        with open(os.path.join(output_dir, 'class_mapping.txt'), 'w', encoding='utf-8') as f:
            for original, mapped in class_mapping.items():
                f.write(f"Original: {original} -> Mapped: {mapped}\n")

        # Convert labels to categorical
        mapped_labels = np.zeros_like(labels)
        for orig_class, new_class in class_mapping.items():
            mapped_labels[labels == orig_class] = new_class

        labels_categorical = tf.keras.utils.to_categorical(mapped_labels, num_classes=n_classes)

        # Split data
        images_train, images_test, labels_train, labels_test = train_test_split(
            images, labels_categorical, test_size=0.2, random_state=42
        )

        # Create and compile model
        print("\nCreating U-Net model...")
        input_shape = (images.shape[1], images.shape[2], len(selected_bands))
        model = create_unet(input_shape, n_classes)

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Save model summary
        with open(os.path.join(output_dir, 'model_summary.txt'), 'w', encoding='utf-8') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        model.summary()

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(output_dir, 'best_model.h5'),
                save_best_only=True,
                monitor='val_loss'
            )
        ]

        # Train model
        print("\nTraining model...")
        history = model.fit(
            images_train,
            labels_train,
            batch_size=8,
            epochs=100,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )

        # Save training history plot
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_history.png'))
        plt.close()

        # Evaluate model
        print("\nEvaluating model...")
        test_loss, test_accuracy = model.evaluate(images_test, labels_test, verbose=1)

        # Save metrics
        with open(os.path.join(output_dir, 'test_metrics.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Test Loss: {test_loss}\n")
            f.write(f"Test Accuracy: {test_accuracy}\n")

        # Generate predictions
        predictions = model.predict(images_test)
        predicted_classes = np.argmax(predictions, axis=-1)
        true_classes = np.argmax(labels_test, axis=-1)

        # Calculate and save classification report
        flat_pred = predicted_classes.flatten()
        flat_true = true_classes.flatten()

        report = classification_report(flat_true, flat_pred)
        print("\nClassification Report:")
        print(report)

        with open(os.path.join(output_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
            f.write(report)

        # Save final model
        model.save(os.path.join(output_dir, 'final_model.h5'))
        print(f"\nResults saved in directory: {output_dir}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()