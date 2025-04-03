# make predictions on the new brain MRI images
import os
import cv2
import numpy as np
import tensorflow as tf
import imutils
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd


class BrainTumorPredictor:
    '''
    Make predictions on new brain MRI images for tumor detection

    '''

    def __init__(self, model_path):
        """
        Initializes a new instance of the BrainTumorPredictor class.

        Args:
            model_path (str): The path to the trained model file.
        """
        self.model_path = model_path
        self.model = None
        self.image_size = (200, 200)  # Size expected by the model

    def load_model(self):
        """
        Load the trained model from the specified path.

        Returns:
            None
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}")

        self.model = tf.keras.models.load_model(self.model_path)
        print(f"Model loaded successfully from {self.model_path}")

    def crop_brain_tumor(self, image):
        """
        Crop the brain MRI image to focus on the relevant area by removing excess black background.

        Args:
            image (numpy.ndarray): The input brain MRI image.

        Returns:
            numpy.ndarray: The cropped image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        thres = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thres = cv2.erode(thres, None, iterations=2)
        thres = cv2.dilate(thres, None, iterations=2)

        cnts = cv2.findContours(
            thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # If no contours are found, return the original image
        if not cnts:
            return image

        c = max(cnts, key=cv2.contourArea)

        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

        # If cropping fails, return the original image
        if new_image.size == 0:
            return image

        return new_image

    def preprocess_image(self, image_path):
        """
        Preprocess a single image for prediction.

        Args:
            image_path (str): Path to the image file.

        Returns:
            numpy.ndarray: Preprocessed image ready for prediction.
        """
        # Check if the file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")

        # Crop the image
        img = self.crop_brain_tumor(img)

        # Resize the image
        img = cv2.resize(img, self.image_size)

        # Convert to grayscale (as the model was trained on grayscale images)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Normalize the image
        img = img / 255.0

        # Reshape for the model (add channel dimension)
        img = np.expand_dims(img, axis=-1)

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        return img

    def load_images_from_directory(self, directory):
        """
        Load all images from a directory for batch prediction.

        Args:
            directory (str): Path to the directory containing images.

        Returns:
            list: List of preprocessed images.
            list: List of image file paths.
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        image_paths = [
            os.path.join(directory, f) for f in os.listdir(directory)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        if not image_paths:
            raise ValueError(f"No images found in {directory}")

        images = []
        valid_paths = []

        for path in image_paths:
            try:
                img = self.preprocess_image(path)
                images.append(img)
                valid_paths.append(path)
            except Exception as e:
                print(f"Error processing {path}: {e}")

        if not images:
            raise ValueError("No valid images could be processed")

        return np.vstack(images), valid_paths

    def predict_single_image(self, image_path):
        """
        Make a prediction on a single image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Prediction result ("TUMOR" or "NO")
            float: Confidence score
        """
        if self.model is None:
            self.load_model()

        img = self.preprocess_image(image_path)

        # Make prediction
        pred = self.model.predict(img)[0]

        # Get the predicted class and confidence
        # Class indices from the training: {'no': 0, 'yes': 1}
        # We need to reverse this to match our output format
        pred_class = "NO" if np.argmax(pred) == 0 else "TUMOR"
        confidence = pred[np.argmax(pred)]

        return pred_class, float(confidence)

    def predict_batch(self, directory):
        """
        Make predictions on all images in a directory.

        Args:
            directory (str): Path to the directory containing images.

        Returns:
            pandas.DataFrame: DataFrame containing image paths and predictions.
        """
        if self.model is None:
            self.load_model()

        images, image_paths = self.load_images_from_directory(directory)

        # Make predictions
        predictions = self.model.predict(images)

        # Process results
        results = []
        for i, pred in enumerate(predictions):
            pred_class = "NO" if np.argmax(pred) == 0 else "TUMOR"
            confidence = float(pred[np.argmax(pred)])
            results.append({
                'image_path': image_paths[i],
                'prediction': pred_class,
                'confidence': confidence
            })

        return pd.DataFrame(results)


if __name__ == "__main__":
    # Setting the default parameters
    model_path = "model_3.keras"

    # Create an instance of the predictor
    predictor = BrainTumorPredictor(model_path)

    # Load the model
    predictor.load_model()

    # To Predict a single image
    try:
        image_path = "./dataset/pred/pred0.jpg"
        prediction, confidence = predictor.predict_single_image(image_path)
        print(f"Image: {image_path}")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.4f}")
    except Exception as e:
        print(f"Error predicting single image: {e}")

    # To Predict all images in a directory
    try:
        predictions_df = predictor.predict_batch("./dataset/pred")
        print("\nBatch Predictions:")
        print(predictions_df)

        # Save predictions to CSV
        predictions_df.to_csv("brain_tumor_predictions.csv", index=False)
        print("Predictions saved to brain_tumor_predictions.csv")
    except Exception as e:
        print(f"Error in batch prediction: {e}")
