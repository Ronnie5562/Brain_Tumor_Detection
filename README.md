# Brain Tumor Classifier using CNN

This project is a Brain Tumor Classifier using Convolutional Neural Networks (CNN), built with FastAPI to expose the model via an API. It helps in classifying brain tumor images as either benign or malignant.

![Video](https://github.com/user-attachments/assets/ca257281-ba0e-498d-a797-d8a135ad4102)

## Project Description

This project is designed to classify brain tumor images using a deep learning model based on Convolutional Neural Networks (CNN). The model is deployed through a FastAPI API, which can be accessed and tested via HTTP requests. The classifier accepts image inputs and returns predictions of the tumor's nature (Tumor, No Tumor).

### Features

- A pre-trained CNN model for brain tumor classification.
- FastAPI to provide an easy-to-use API for predictions.
- Simple and fast deployment with minimal setup.

## Video Demo

You can watch the video demonstration of the project on YouTube:
[Watch the Demo](https://www.youtube.com/link_to_demo)

## Requirements

Before running the project, you need to install the necessary dependencies. These are listed in the `requirements.txt` file.

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Ronnie5562/Brain_Tumor_Detection.git
cd Brain_Tumor_Detection
```

### 2. Create a virtual environment (optional but recommended)
For Python 3.x, you can create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
```

### 3.  Install the required dependencies
Install the required Python libraries using pip:

```bash
pip install -r requirements.txt
```

### 4. Run the API
Run the FastAPI server by executing:

```bash
python main.py
```

### API Documentation
You can find detailed API documentation at:

```bash
http://127.0.0.1:8000/docs
```


### Frontend Repo

- [View Here](https://github.com/Ronnie5562/CORTEX_AI_FRONTEND)

### Deployed Frontend

- [View Here](https://cortex-ai-frontend.vercel.app/)
