# Plant Disease Classification with TensorFlow

## Overview

This project involves classifying plant diseases using a Convolutional Neural Network (CNN) with TensorFlow. The focus is on potato plant diseases, specifically Early Blight, Late Blight, and Healthy states. The model is trained on the PlantVillage dataset and is integrated into a simple web application for demonstration purposes.

## Table of Contents

- [Project Description](#project-description)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Web Application](#web-application)
- [Usage](#usage)
- [Future Work](#future-work)
- [License](#license)
- [Contact](#contact)

## Project Description

This project aims to classify potato plant diseases using a deep learning model. The model predicts whether a potato plant is suffering from Early Blight, Late Blight, or is Healthy. A simple web application has been developed to interact with the trained model, allowing users to test its predictions locally.

## Dataset

- **Source:** [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Description:** Contains images of potato plant leaves categorized into Early Blight, Late Blight, and Healthy classes.
- **Number of Images:** Over 2,000

## Model Architecture

- **Type:** Convolutional Neural Network (CNN)
- **Layers:**
  - Multiple convolutional and pooling layers
  - Fully connected layers
- **Techniques:**
  - Data Augmentation: Horizontal and vertical flips, random rotations
  - Resizing: Uniform input dimensions
  - Normalization: Pixel values scaled to [0, 1]

## Training and Evaluation

- **Training Accuracy:** 98.6%
- **Training Loss:** 0.0278
- **Epochs:** 50
- **Evaluation:** The model was evaluated on a test set to assess its classification performance. Visualizations of training and validation accuracy/loss are included.

## Web Application

- **Server:** FastAPI
- **Frontend:** HTML/CSS
- **Description:** A simple web application was created to test the model. It allows users to upload images and view predictions locally.

## Usage

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/plant-disease-classification.git
   cd plant-disease-classification
