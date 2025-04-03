import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import datetime
import shutil
import pandas as pd


class ModelRetrainer:
    """Helper class for model retraining"""

    def __init__(self, base_model_path, retrain_data_dir, output_dir):
        self.base_model_path = base_model_path
        self.retrain_data_dir = retrain_data_dir
        self.output_dir = output_dir

    def retrain_model(self, epochs=10, batch_size=16, learning_rate=0.0001):
        """Retrain the model with new data"""
        base_model = tf.keras.models.load_model(self.base_model_path)

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2,
        )

        train_generator = train_datagen.flow_from_directory(
            self.retrain_data_dir,
            target_size=(200, 200),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            color_mode='grayscale'
        )

        validation_generator = train_datagen.flow_from_directory(
            self.retrain_data_dir,
            target_size=(200, 200),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            color_mode='grayscale'
        )

        # Set up callbacks
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_path = os.path.join(
            self.output_dir, f"retrained_model_{timestamp}.keras")

        callbacks = [
            ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]

        base_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        history = base_model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks
        )

        history_df = pd.DataFrame(history.history)
        history_df.to_csv(os.path.join(
            self.output_dir, f"training_history_{timestamp}.csv"))

        # Return paths to the retrained model and training history
        return {
            'model_path': model_path,
            'history_path': os.path.join(self.output_dir, f"training_history_{timestamp}.csv"),
            'final_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'training_samples': train_generator.samples,
            'validation_samples': validation_generator.samples
        }
