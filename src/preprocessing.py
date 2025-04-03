import os
import cv2
import imutils
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataPreprocessor:
    """
    A class to handle brain tumor image processing and dataset preparation.
    """

    def __init__(self, base_path="./dataset", target_size=(200, 200), batch_size=20):
        """
        Initialize the processor with paths and parameters.

        Args:
            base_path (str): Base directory containing tumor image folders
            target_size (tuple): Target size for processed images
            batch_size (int): Batch size for data generators
        """
        self.base_path = base_path
        self.target_size = target_size
        self.batch_size = batch_size
        self.train_data = None
        self.test_data = None
        self.train_set = None
        self.validation_set = None
        self.test_set = None

    def load_image_paths(self, category):
        """
        Load all image file paths from the given category folder.

        Args:
            category (str): Subfolder name ('yes' or 'no')

        Returns:
            list: Paths to all JPG images in the folder
        """
        folder_path = os.path.join(self.base_path, category)
        return list(Path(folder_path).glob("*.jpg"))

    def create_dataframe(self, image_paths, label):
        """
        Create a DataFrame with image paths and corresponding labels.

        Args:
            image_paths (list): List of image file paths
            label (str): Label for the images

        Returns:
            pd.DataFrame: DataFrame with image paths and labels
        """
        labels = [label] * len(image_paths)
        return pd.DataFrame({"JPG": list(map(str, image_paths)), "TUMOR_CATEGORY": labels})

    @staticmethod
    def crop_brain_tumor(image):
        """
        Crop the brain tumor region from the given image.

        Args:
            image (numpy.ndarray): Input image

        Returns:
            numpy.ndarray: Cropped image containing tumor region
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thres = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thres = cv2.erode(thres, None, iterations=2)
        thres = cv2.dilate(thres, None, iterations=2)
        cnts = cv2.findContours(
            thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        if not cnts:
            return image  # Return original if no contours found

        c = max(cnts, key=cv2.contourArea)
        extTop, extBot = tuple(c[c[:, :, 1].argmin()][0]), tuple(
            c[c[:, :, 1].argmax()][0])
        extLeft, extRight = tuple(c[c[:, :, 0].argmin()][0]), tuple(
            c[c[:, :, 0].argmax()][0])

        return image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    def preprocess_images(self, image_paths):
        """
        Preprocess images by cropping and saving them back.

        Args:
            image_paths (list): List of image file paths to preprocess
        """
        for filename in image_paths:
            img = cv2.imread(filename)
            if img is not None:
                img = self.crop_brain_tumor(img)
                cv2.imwrite(filename, img)

    def create_image_generators(self):
        """
        Create ImageDataGenerators for training, validation, and test sets.

        Returns:
            tuple: (train_set, validation_set, test_set) generators
        """
        data_gen = ImageDataGenerator(
            rescale=1./255,
            brightness_range=[0.3, 0.9],
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
            validation_split=0.1
        )

        train_set = data_gen.flow_from_dataframe(
            dataframe=self.train_data,
            x_col="JPG",
            y_col="TUMOR_CATEGORY",
            color_mode="grayscale",
            class_mode="categorical",
            subset="training",
            batch_size=self.batch_size,
            target_size=self.target_size
        )

        validation_set = data_gen.flow_from_dataframe(
            dataframe=self.train_data,
            x_col="JPG",
            y_col="TUMOR_CATEGORY",
            color_mode="grayscale",
            class_mode="categorical",
            subset="validation",
            batch_size=self.batch_size,
            target_size=self.target_size
        )

        test_set = data_gen.flow_from_dataframe(
            dataframe=self.test_data,
            x_col="JPG",
            y_col="TUMOR_CATEGORY",
            color_mode="grayscale",
            class_mode="categorical",
            batch_size=self.batch_size,
            target_size=self.target_size
        )

        return train_set, validation_set, test_set

    def prepare_dataset(self):
        """
        Prepare the complete dataset by loading images, preprocessing, and creating generators.

        Returns:
            tuple: (train_set, validation_set, test_set) generators
        """
        # Load image paths
        no_tumor_images = self.load_image_paths("no")
        yes_tumor_images = self.load_image_paths("yes")

        # Create dataframes
        data_no = self.create_dataframe(no_tumor_images, "non-tumorous")
        data_yes = self.create_dataframe(yes_tumor_images, "tumorous")

        # Combine and shuffle
        main_data = pd.concat([data_no, data_yes]).sample(
            frac=1).reset_index(drop=True)

        # Preprocess images
        self.preprocess_images(main_data["JPG"].tolist())

        # Split data
        self.train_data, self.test_data = train_test_split(
            main_data, train_size=0.9, random_state=42)

        # Create generators
        self.train_set, self.validation_set, self.test_set = self.create_image_generators()

        return self.train_set, self.validation_set, self.test_set



if __name__ == "__main__":
    processor = DataPreprocessor(
        base_path="./dataset",
        target_size=(200, 200),
        batch_size=20
    )
    train_set, validation_set, test_set = processor.prepare_dataset()

    print("Data preprocessing complete.")
    print(f"Train set size: {len(train_set)}")
    print(f"Validation set size: {len(validation_set)}")
    print(f"Test set size: {len(test_set)}")
