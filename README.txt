AI-Powered Multiple Disease Prediction System


Overview

This project involves the development of AI-powered predictive models capable of assessing the risk for multiple diseases based on user-provided health data. The primary diseases targeted by this project include:

	- Diabetes
	- Heart disease
	- Parkinson’s disease

The predictive models are integrated into a streamlined, interactive web application built using Streamlit, providing users with a user-friendly and intuitive interface.



Features

Disease Risk Prediction

	1. Heart Disease Prediction: Estimates the risk of heart disease using various health metrics.

	2. Parkinson's Disease Prediction: Assesses the likelihood of developing Parkinson’s disease through specific medical data points.

	3. Diabetes Prediction: Predicts the likelihood of diabetes based on health-related parameters.


Getting Started

Prerequisites
	Python 3.7 or higher
	Pip (Python package installer)

Installation
Clone the Repository:
	git clone https://github.com/yourusername/AI-Multiple-Disease-Predictor.git
	cd AI-Multiple-Disease-Predictor

Set Up a Virtual Environment: On Windows:
	python -m venv venv
	venv\Scripts\activate

On macOS/Linux:
	python3 -m venv venv
	source venv/bin/activate

Install Dependencies:
	pip install -r requirements.txt

Run the Application:
	streamlit run app.py

Navigate to http://localhost:8501 in your browser to access the disease prediction interface.



Models and Architecture

1. Heart Disease Model: Pre-trained model file heart_disease_model.sav used to predict heart disease risk.

2. Parkinson’s Disease Model: Model file parkinsons_model.sav for predicting Parkinson's disease.

3. Diabetes Model: Model file diabetes_model.sav used for assessing diabetes risk.

Each model was trained using supervised learning techniques on relevant medical datasets.



Data

Datasets used for training and predictions:

	- Datasets for Heart Disease, Parkinson’s Disease, and Diabetes are in the csv files labeled 	diabetes.csv, heart_disease_data.csv, and parkinsons.csv in the datasets folder

	- Ensure datasets are correctly structured and available within the specified directory to 	ensure proper functionality.


Project Structure
colab_files_to_train_models/
│
├── diabetes.ipynb
├── heart.ipynb
├── parkinsons.ipynb
├── datasets/
├── app.py
├── requirements.txt
├── heart_disease_model.sav
├── parkinsons_model.sav
├── diabetes_model.sav
└── README.md


Contributors

	- Ibrima K Njie

