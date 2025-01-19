import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway


class EDA:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def analyze_band_correlations(self):
        """
        Analyze the correlation between the bands.
        """
        concatenated_bands = self.concatenate_bands()
        corr_matrix = np.corrcoef(concatenated_bands)

        self.plot_correlation_matrix(corr_matrix)

    def analyze_class_correlations(self):
        """
        Analyze the correlation between the classes and the bands.
        """
        unique_classes, counts = self.get_unique_classes()

        for class_index in unique_classes:
            pixels_class = self.extract_pixels_for_class(class_index)
            mean_class = np.mean(pixels_class, axis=0)
            std_class = np.std(pixels_class, axis=0)

            self.plot_mean_intensity_for_class(class_index, mean_class)

    def perform_anova(self):
        """
        Perform ANOVA to analyze the statistical significance of the differences between classes for each band.
        """
        labels_flattened = np.concatenate([lbl.flatten() for lbl in self.labels])
        unique_classes = np.unique(labels_flattened)

        num_bands = self.images[0].shape[2]
        results = {}

        for band_index in range(num_bands):
            band_flattened = np.concatenate([img[:, :, band_index].flatten() for img in self.images])
            band_data_by_class = [band_flattened[labels_flattened == class_value] for class_value in unique_classes]
            f_value, p_value = f_oneway(*band_data_by_class)
            results[band_index] = (f_value, p_value)
            print(f"Band {band_index} ANOVA F-value: {f_value}, p-value: {p_value}")

        return results

    def concatenate_bands(self):
        """
        Concatenate the flattened bands across all images.
        """
        num_bands = self.images[0].shape[2]
        concatenated_bands = []

        for band in range(num_bands):
            band_data = [img[:, :, band].flatten() for img in self.images]
            concatenated_band = np.concatenate(band_data, axis=0)
            concatenated_bands.append(concatenated_band)

        return np.array(concatenated_bands)

    def concatenate_labels(self):
        """
        Concatenate the labels across all images.
        """
        concatenated_labels = []
        for label in self.labels:
            label_data = label.flatten()
            concatenated_labels.append(label_data)

        concatenated_labels = np.concatenate(concatenated_labels, axis=0)
        # Reshape to a single row
        concatenated_labels = concatenated_labels.reshape(1, -1)

        return concatenated_labels

    def get_unique_classes(self):
        """
        Get the list of unique classes and their counts.
        """
        labels_flattened = np.concatenate([lbl.flatten() for lbl in self.labels])
        unique_classes, counts = np.unique(labels_flattened, return_counts=True)
        return unique_classes, counts

    def extract_pixels_for_class(self, class_index):
        """
        Extract the pixel values for a given class.
        """
        pixels_class = []
        for img, lbl in zip(self.images, self.labels):
            pixels = img[lbl == class_index]
            pixels_class.append(pixels)
        return np.concatenate(pixels_class, axis=0)

    def plot_correlation_matrix(self, corr_matrix):
        """
        Plot a heatmap of the correlation matrix.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=0.5, vmax=1)
        plt.title("Correlation Matrix Heatmap")
        plt.show()

    def plot_mean_intensity_for_class(self, class_index, mean_class):
        """
        Plot the mean intensity of each band for a given class.
        """
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(mean_class)), mean_class)
        plt.xlabel('Bands')
        plt.ylabel('Mean Intensity')
        plt.title(f'Mean Intensity of Each Band for Class {class_index}')
        plt.show()




