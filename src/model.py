import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense


class BrainTumorModel:
    """
    A class to handle the brain tumor classification model.
    Includes model definition, training, evaluation, and visualization.
    """

    def __init__(self, model_dir="./models", input_shape=(200, 200, 1), num_classes=2):
        """
        Initialize the model.

        Args:
            model_dir (str): Directory to save models
            input_shape (tuple): Shape of input images (height, width, channels)
            num_classes (int): Number of output classes
        """
        self.model_dir = model_dir
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.predictions = None

        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)

    def build_model(self):
        """
        Build the CNN model architecture.

        Returns:
            tensorflow.keras.models.Sequential: Compiled model
        """
        model = Sequential()

        # First convolutional block
        model.add(Conv2D(32, (5, 5), activation="relu",
                  input_shape=self.input_shape))
        model.add(MaxPool2D((2, 2)))
        model.add(Dropout(0.2))

        # Second convolutional block
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPool2D((2, 2)))
        model.add(Dropout(0.2))

        # Third convolutional block
        model.add(Conv2D(128, (3, 3), activation="relu"))
        model.add(MaxPool2D((2, 2)))
        model.add(Dropout(0.2))

        # Fourth convolutional block
        model.add(Conv2D(256, (3, 3), activation="relu"))
        model.add(MaxPool2D((2, 2)))
        model.add(Dropout(0.2))

        # Flatten and dense layers
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(512, activation="relu"))
        model.add(Dense(self.num_classes, activation="softmax"))

        # Compile model
        model.compile(
            optimizer=RMSprop(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        self.model = model
        return model

    def train(self, train_set, validation_set, epochs=50, steps_per_epoch=120, use_early_stopping=True):
        """
        Train the model on the provided data.

        Args:
            train_set: Training data generator
            validation_set: Validation data generator
            epochs (int): Maximum number of training epochs
            steps_per_epoch (int): Number of steps per epoch
            use_early_stopping (bool): Whether to use early stopping

        Returns:
            tensorflow.keras.callbacks.History: Training history
        """
        callbacks = []

        if use_early_stopping:
            early_stopping = EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=10,
                verbose=1
            )
            callbacks.append(early_stopping)

        history = self.model.fit(
            train_set,
            validation_data=validation_set,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks
        )

        self.history = history
        return history

    def evaluate(self, test_set):
        """
        Evaluate the model on test data.

        Args:
            test_set: Test data generator

        Returns:
            list: Evaluation metrics [loss, accuracy]
        """
        results = self.model.evaluate(test_set, verbose=1)
        return results

    def predict(self, test_set):
        """
        Make predictions on test data.

        Args:
            test_set: Test data generator

        Returns:
            numpy.ndarray: Predicted class indices
        """
        predictions = self.model.predict(test_set)
        self.predictions = predictions.argmax(axis=-1)
        return self.predictions

    def get_prediction_labels(self, class_indices=None):
        """
        Convert numerical predictions to class labels.

        Args:
            class_indices (dict): Dictionary mapping class indices to class names

        Returns:
            list: Class label for each prediction
        """
        if class_indices is None:
            class_indices = {0: "TUMOR", 1: "NO"}

        # Invert the dictionary to map indices to labels
        idx_to_class = {v: k for k, v in class_indices.items()}

        # Convert numerical predictions to class labels
        labels = [idx_to_class[pred] for pred in self.predictions]
        return labels

    def plot_confusion_matrix(self, y_true, y_pred, classes=None):
        """
        Plot the confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            classes (list): List of class names

        Returns:
            matplotlib.figure.Figure: The confusion matrix plot
        """
        if classes is None:
            classes = ["TUMOR", "NO"]

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        # Save the plot
        plt.savefig(os.path.join(self.model_dir, 'confusion_matrix.png'))
        return plt.gcf()

    def plot_training_history(self):
        """
        Plot the training history.

        Returns:
            matplotlib.figure.Figure: The training history plot
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return None

        history_dict = self.history.history

        # Create a figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot accuracy
        axes[0].plot(history_dict['accuracy'], label='Training')
        axes[0].plot(history_dict['val_accuracy'], label='Validation')
        axes[0].set_title('Model Accuracy')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()

        # Plot loss
        axes[1].plot(history_dict['loss'], label='Training')
        axes[1].plot(history_dict['val_loss'], label='Validation')
        axes[1].set_title('Model Loss')
        axes[1].set_ylabel('Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()

        plt.tight_layout()

        # Save the plot
        plt.savefig(os.path.join(self.model_dir, 'training_history.png'))
        return fig

    def save_model(self, custom_name=None):
        """
        Save the trained model.

        Args:
            custom_name (str): Optional custom name for the model

        Returns:
            str: Path to the saved model
        """
        if self.model is None:
            print("No model to save. Build and train the model first.")
            return None

        if custom_name:
            model_path = os.path.join(self.model_dir, f"{custom_name}.keras")
        else:
            # Find the next available model number
            existing_models = [f for f in os.listdir(self.model_dir)
                               if f.startswith('model_') and f.endswith('.keras')]
            model_numbers = [int(f.split('_')[1].split('.')[0]) for f in existing_models
                             if f.split('_')[1].split('.')[0].isdigit()]
            next_number = max(model_numbers, default=0) + 1
            model_path = os.path.join(
                self.model_dir, f"model_{next_number}.keras")

        self.model.save(model_path)
        print(f"Model saved to {model_path}")
        return model_path

    def load_model(self, model_path):
        """
        Load a saved model.

        Args:
            model_path (str): Path to the saved model

        Returns:
            tensorflow.keras.models.Sequential: Loaded model
        """
        self.model = tf.keras.models.load_model(model_path)
        return self.model

    def print_summary(self):
        """
        Print the model summary.

        Returns:
            None
        """
        if self.model is None:
            print("No model available. Build the model first.")
            return

        self.model.summary()


# Example usage
if __name__ == "__main__":
    model = BrainTumorModel(model_dir="./models")

    model.build_model()

    model.print_summary()

    print("Model built successfully. To train the model, provide data generators.")
