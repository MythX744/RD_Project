import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import rasterio
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from datetime import datetime
import json


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


# Configuration
class Config:
    # Image properties
    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 128
    NUM_BANDS = None
    NUM_CLASSES = 34

    # Training parameters
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 1e-4

    # Selected bands will be stored here
    SELECTED_BANDS = None


def create_deeplabv3(input_shape, num_classes):
    """Create DeepLabV3 model"""

    # Model architecture remains the same
    def conv_bn_relu(x, filters, kernel_size=3, strides=1, dilation_rate=1):
        x = tf.keras.layers.Conv2D(
            filters, kernel_size,
            strides=strides,
            padding='same',
            dilation_rate=dilation_rate
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        return x

    def atrous_spatial_pyramid_pooling(x):
        """ASPP with different dilation rates"""
        conv_1x1 = conv_bn_relu(x, 256, kernel_size=1)
        conv_3x3_1 = conv_bn_relu(x, 256, kernel_size=3, dilation_rate=6)
        conv_3x3_2 = conv_bn_relu(x, 256, kernel_size=3, dilation_rate=12)
        conv_3x3_3 = conv_bn_relu(x, 256, kernel_size=3, dilation_rate=18)

        global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()(x)
        global_avg_pool = tf.keras.layers.Reshape((1, 1, x.shape[-1]))(global_avg_pool)
        global_avg_pool = conv_bn_relu(global_avg_pool, 256, kernel_size=1)
        global_avg_pool = tf.keras.layers.UpSampling2D(
            size=(x.shape[1], x.shape[2]),
            interpolation='bilinear'
        )(global_avg_pool)

        x = tf.keras.layers.Concatenate()(
            [conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, global_avg_pool]
        )

        return x

    inputs = tf.keras.Input(shape=input_shape)

    x = conv_bn_relu(inputs, 64, strides=2)
    x = conv_bn_relu(x, 64)
    x = conv_bn_relu(x, 128)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = conv_bn_relu(x, 256)
    x = conv_bn_relu(x, 256)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = conv_bn_relu(x, 512)
    x = conv_bn_relu(x, 512)

    x = atrous_spatial_pyramid_pooling(x)
    x = conv_bn_relu(x, 256, kernel_size=1)

    x = tf.keras.layers.UpSampling2D(
        size=(8, 8),
        interpolation='bilinear'
    )(x)

    outputs = tf.keras.layers.Conv2D(
        num_classes,
        kernel_size=1,
        activation='softmax'
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def load_initial_images(image_dir):
    """Load all images initially to select bands"""
    image_files = sorted(os.listdir(image_dir))
    images = []

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        with rasterio.open(image_path) as src:
            image = src.read()
            image = np.transpose(image, (1, 2, 0))
            images.append(image)

    return np.array(images)


def load_and_preprocess_image(image_path, selected_bands):
    """Load and preprocess a single hyperspectral image with selected bands"""
    with rasterio.open(image_path) as src:
        # Read only the selected bands
        image = src.read(selected_bands)
        image = np.transpose(image, (1, 2, 0))

    # Normalize image
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    std = np.std(image, axis=(0, 1), keepdims=True) + 1e-10
    image = (image - mean) / std

    return image


def load_and_preprocess_mask(mask_path, class_mapping):
    """Load and preprocess a single mask"""
    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    mapped_mask = np.zeros_like(mask)
    for orig_class, new_class in class_mapping.items():
        mapped_mask[mask == orig_class] = new_class

    mask_onehot = tf.keras.utils.to_categorical(mapped_mask, num_classes=34)
    return mask_onehot


def load_dataset(image_dir, mask_dir, class_mapping, config):
    """Load and preprocess all images and masks in the dataset."""
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    # First, load a sample of images to select bands
    print("Loading initial images for band selection...")
    initial_images = load_initial_images(image_dir)

    # Select bands
    print("Selecting optimal bands...")
    config.SELECTED_BANDS = select_bands(initial_images)
    config.NUM_BANDS = len(config.SELECTED_BANDS)
    print(f"Selected {config.NUM_BANDS} bands: {config.SELECTED_BANDS}")

    # Now load only selected bands for all images
    print("Loading selected bands for all images...")
    images = []
    masks = []

    # Convert SELECTED_BANDS to list of band indices (adding 1 because rasterio bands are 1-indexed)
    selected_bands = [band + 1 for band in config.SELECTED_BANDS]

    for image_file, mask_file in zip(image_files, mask_files):
        # Load and preprocess image with selected bands
        image_path = os.path.join(image_dir, image_file)
        image = load_and_preprocess_image(image_path, selected_bands)
        images.append(image)

        # Load and preprocess mask
        mask_path = os.path.join(mask_dir, mask_file)
        mask = load_and_preprocess_mask(mask_path, class_mapping)
        masks.append(mask)

    return np.array(images), np.array(masks)


def train_test_val_split(images, masks, test_size=0.2, val_size=0.1, random_state=42):
    """Split the dataset into train, validation, and test sets."""
    images_train, images_test, masks_train, masks_test = train_test_split(
        images, masks, test_size=test_size, random_state=random_state
    )

    val_size_adjusted = val_size / (1 - test_size)
    images_train, images_val, masks_train, masks_val = train_test_split(
        images_train, masks_train, test_size=val_size_adjusted, random_state=random_state
    )

    return images_train, images_val, images_test, masks_train, masks_val, masks_test


def save_model_outputs(model, history, predictions_test, masks_test, f1, config, test_loss=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"deeplabv3_defcoef_{timestamp}"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Save band selection information
    with open(os.path.join(folder_name, 'selected_bands.txt'), 'w') as f:
        f.write(f"Automatically selected bands: {config.SELECTED_BANDS}\n")
        f.write(f"Number of bands used: {config.NUM_BANDS}\n")

    # Save model summary
    with open(os.path.join(folder_name, 'model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Save model
    model.save(os.path.join(folder_name, 'model.h5'))

    # Save classification report
    pred_classes = np.argmax(predictions_test, axis=-1).flatten()
    true_classes = np.argmax(masks_test, axis=-1).flatten()
    class_report = classification_report(true_classes, pred_classes)
    with open(os.path.join(folder_name, 'classification_report.txt'), 'w') as f:
        f.write(class_report)

    # Save metrics
    metrics = {
        'f1_score': float(f1),
        'test_accuracy': float(history.history['val_accuracy'][-1]),
        'test_loss': float(history.history['val_loss'][-1])
    }

    with open(os.path.join(folder_name, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # Save test metrics
    test_metrics = {
        'test_f1_score': float(f1),
        'test_loss': float(test_loss) if test_loss is not None else None
    }

    with open(os.path.join(folder_name, 'test_set_metrics.txt'), 'w') as f:
        f.write("Test Set Metrics:\n")
        f.write("-----------------\n")
        f.write(f"Test F1 Score: {test_metrics['test_f1_score']:.4f}\n")
        if test_loss is not None:
            f.write(f"Test Loss: {test_metrics['test_loss']:.4f}\n")

    # Save training history plot
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, 'training_history.png'))
    plt.close()

    print(f"All outputs saved in folder: {folder_name}")


def train_on_dataset():
    config = Config()

    class_mapping = {
        10: 0, 11: 1, 12: 2, 20: 3, 51: 4, 52: 5, 61: 6, 62: 7, 71: 8, 72: 9,
        81: 10, 82: 11, 91: 12, 120: 13, 121: 14, 122: 15, 130: 16, 140: 17,
        150: 18, 152: 19, 153: 20, 181: 21, 182: 22, 183: 23, 184: 24, 185: 25,
        186: 26, 187: 27, 190: 28, 200: 29, 201: 30, 202: 31, 210: 32, 220: 33
    }

    image_dir = 'C:/Users/malak/Documents/CV_project/Code/malak_images'
    mask_dir = 'C:/Users/malak/Documents/CV_project/Code/malak_landcover'

    # Load dataset with automatic band selection
    images, masks = load_dataset(image_dir, mask_dir, class_mapping, config)

    # Split dataset
    images_train, images_val, images_test, masks_train, masks_val, masks_test = train_test_val_split(
        images, masks, test_size=0.2, val_size=0.1
    )

    # Create and compile model
    model = create_deeplabv3(
        input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.NUM_BANDS),
        num_classes=config.NUM_CLASSES
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    history = model.fit(
        images_train, masks_train,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(images_val, masks_val),
        verbose=1
    )

    # Get predictions
    predictions_test = model.predict(images_test, verbose=1)

    # Calculate F1 score
    pred_classes = np.argmax(predictions_test, axis=-1).flatten()
    true_classes = np.argmax(masks_test, axis=-1).flatten()
    f1 = f1_score(true_classes, pred_classes, average='weighted')

    # Evaluate on test set
    test_loss = model.evaluate(images_test, masks_test, verbose=0)[0]
    print(f"Test Loss: {test_loss}")

    # Save all outputs
    save_model_outputs(model, history, predictions_test, masks_test, f1, config, test_loss)

    return model, history, f1


if __name__ == '__main__':
    model, history, f1 = train_on_dataset()