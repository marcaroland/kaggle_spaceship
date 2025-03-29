# Spaceship Titanic - Machine Learning Model

- This project builds a machine learning pipeline to predict the Transported status of passengers in the Spaceship Titanic Kaggle competition. It preprocesses data, performs feature engineering, optimizes multiple machine learning models using Optuna, and combines them into a VotingClassifier for final predictions.

# Code Workflow
- 1️⃣ Load Data
    - The script loads train.csv and test.csv using pandas.

- 2️⃣ Feature Engineering
    - The FeatureEngineering class handles:

    - Missing value imputation (e.g., replacing missing values with median values)

    - Feature transformations (e.g., one-hot encoding categorical features)

- 3️⃣ Model Training and Hyperparameter Tuning
The Modelling class:

    - Optimizes models (e.g., RandomForest, XGBoost, LightGBM) using Optuna

    - Trains an ensemble model using VotingClassifier

- 4️⃣ Predictions & Submission
    - Once trained, the ensemble model predicts the Transported status for the test dataset.