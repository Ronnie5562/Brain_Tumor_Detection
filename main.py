import os
import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import shutil
import uuid
from typing import List
import sys
from pathlib import Path

from src.prediction import BrainTumorPredictor
from src.retrain import ModelRetrainer


app = FastAPI(
    title="Brain Tumor Detection API",
    description="API for detecting brain tumors from MRI images",
    version="1.0.0"
)

# Directories
MODEL_DIR = './models'
UPLOAD_DIR = './uploads'
TEMP_DIR = './uploads/temp'
MODEL_PATH = './models/model_2.keras'

RETRAIN_MODELS_DIR = './models/retrained'
RETRAIN_DATA_DIR = './data/retrain_data'


# Ensure upload directories exist
Path(UPLOAD_DIR).mkdir(exist_ok=True)
Path(TEMP_DIR).mkdir(exist_ok=True)
Path(RETRAIN_MODELS_DIR).mkdir(exist_ok=True)

# Create predictor instance
predictor = BrainTumorPredictor(model_path=MODEL_PATH)


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    filename: str
    prediction: str
    confidence: float


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse]


# Configure CORS
origins = [
    "http://localhost:5173",
    "http://localhost",
    "http://localhost:8080",
    "https://cortex-ai-frontend.vercel.app/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        predictor.load_model()
    except Exception as e:
        print(f"Error loading model: {e}")


@app.get("/")
def root():
    """Root endpoint to check if API is running"""
    return {"message": "Brain Tumor Detection API is running"}


@app.post("/predict/", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """
    Predict if a brain MRI image contains a tumor
    """
    try:
        # Generate a unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_location = os.path.join(TEMP_DIR, unique_filename)

        # Save the uploaded file
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Make prediction
        prediction, confidence = predictor.predict_single_image(file_location)

        # Move file to permanent storage if needed
        permanent_location = os.path.join(UPLOAD_DIR, unique_filename)
        shutil.move(file_location, permanent_location)

        return {
            "filename": file.filename,
            "prediction": prediction,
            "confidence": confidence
        }

    except Exception as e:
        # Clean up temp file if it exists
        if 'file_location' in locals() and os.path.exists(file_location):
            os.remove(file_location)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain/")
async def retrain_model(background_tasks: BackgroundTasks,
                        epochs: int = 10,
                        batch_size: int = 16,
                        learning_rate: float = 0.0001):
    """
    Retrain the brain tumor detection model with new data
    """
    try:
        # Verify that training data exists
        yes_dir = os.path.join(RETRAIN_DATA_DIR, 'yes')
        no_dir = os.path.join(RETRAIN_DATA_DIR, 'no')

        if not os.path.exists(yes_dir) or not os.path.exists(no_dir):
            raise HTTPException(
                status_code=400,
                detail=f"Training data directories not found at {yes_dir} and {no_dir}"
            )

        # Count images in training directories
        yes_images = len([f for f in os.listdir(yes_dir)
                         if os.path.isfile(os.path.join(yes_dir, f))])
        no_images = len([f for f in os.listdir(no_dir)
                        if os.path.isfile(os.path.join(no_dir, f))])

        if yes_images == 0 or no_images == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Training data missing: Found {yes_images} 'yes' images and {no_images} 'no' images"
            )

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        job_dir = os.path.join(RETRAIN_MODELS_DIR, timestamp)
        os.makedirs(job_dir, exist_ok=True)

        # Initialize the retrainer
        retrainer = ModelRetrainer(
            base_model_path=MODEL_PATH,
            retrain_data_dir=RETRAIN_DATA_DIR,
            output_dir=job_dir
        )

        # Run retraining as a background task
        def retrain_task():
            try:
                result = retrainer.retrain_model(
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate
                )

                retrained_model_path = result['model_path']
                backup_path = f"{MODEL_PATH}.backup.{timestamp}"

                shutil.copy(MODEL_PATH, backup_path)

                shutil.copy(retrained_model_path, MODEL_PATH)

                predictor.load_model()

                # Save retraining metadata
                metadata = {
                    "timestamp": timestamp,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "yes_images": yes_images,
                    "no_images": no_images,
                    "training_samples": result['training_samples'],
                    "validation_samples": result['validation_samples'],
                    "final_accuracy": result['final_accuracy'],
                    "final_val_accuracy": result['final_val_accuracy'],
                    "model_backup_path": backup_path,
                    "new_model_path": retrained_model_path
                }

                pd.DataFrame([metadata]).to_csv(os.path.join(
                    job_dir, "retraining_metadata.csv"), index=False)

            except Exception as e:
                print(f"Retraining failed: {str(e)}")

        # Start the background task
        background_tasks.add_task(retrain_task)

        return JSONResponse(content={
            "message": "Retraining started in the background",
            "job_id": timestamp,
            "training_data": {
                "yes_images": yes_images,
                "no_images": no_images
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-batch/", response_model=BatchPredictionResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict brain tumors for multiple MRI images
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    predictions = []
    temp_files = []

    try:
        # Save all files to temp directory
        for file in files:
            file_extension = os.path.splitext(file.filename)[1]
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_location = os.path.join(TEMP_DIR, unique_filename)
            temp_files.append(file_location)

            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        # Process each file
        for i, temp_file in enumerate(temp_files):
            prediction, confidence = predictor.predict_single_image(temp_file)

            # Move to permanent storage
            permanent_location = os.path.join(
                UPLOAD_DIR, os.path.basename(temp_file))
            shutil.move(temp_file, permanent_location)

            predictions.append({
                "filename": files[i].filename,
                "prediction": prediction,
                "confidence": confidence
            })

        return {"predictions": predictions}

    except Exception as e:
        # Clean up any temp files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-and-predict-directory/")
async def upload_and_predict_directory(batch_name: str = Form(...), files: List[UploadFile] = File(...)):
    """
    Upload multiple files to a directory and run batch prediction
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # Create a directory for this batch
    batch_dir = os.path.join(UPLOAD_DIR, batch_name)
    os.makedirs(batch_dir, exist_ok=True)

    try:
        # Save all files to the batch directory
        for file in files:
            file_location = os.path.join(batch_dir, file.filename)
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        # Run batch prediction on the directory
        predictions_df = predictor.predict_batch(batch_dir)

        # Save results to CSV
        csv_path = os.path.join(batch_dir, f"{batch_name}_predictions.csv")
        predictions_df.to_csv(csv_path, index=False)

        # Return results as JSON
        return JSONResponse(content={
            "message": f"Batch {batch_name} processed successfully",
            "predictions": predictions_df.to_dict(orient="records"),
            "results_csv": csv_path
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download-results/{batch_name}")
async def download_results(batch_name: str):
    """
    Download the CSV results for a batch prediction
    """
    csv_path = os.path.join(UPLOAD_DIR, batch_name,
                            f"{batch_name}_predictions.csv")

    if not os.path.exists(csv_path):
        raise HTTPException(
            status_code=404, detail=f"Results for batch {batch_name} not found")

    return FileResponse(
        path=csv_path,
        filename=f"{batch_name}_predictions.csv",
        media_type="text/csv"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
