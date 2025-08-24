# diabetes_prediction
Diabetes Prediction using Machine Learning
This project aims to predict the likelihood of a person having diabetes based on various health-related features. The project uses a balanced dataset and employs several machine learning models to achieve accurate predictions. This README provides a comprehensive overview of the project, including data analysis, modeling, and evaluation.

Table of Contents
Project Overview

Data Analysis

Data Preparation

Modeling

Evaluation

How to Use

Project Overview
The primary goal of this project is to build a machine learning model that can accurately predict whether a person has diabetes. The project uses a dataset containing various patient attributes, such as age, BMI, blood glucose level, and smoking history. The analysis involves several stages, including data cleaning, exploratory data analysis, feature engineering, and model training and evaluation.

Data Analysis
The initial phase of the project involved a thorough analysis of the dataset to understand its characteristics and identify any potential issues.

Dataset Shape and Head
The dataset consists of 183,000 samples and 9 features. A quick look at the first few rows of the data gives an idea of the features and their values.

Data Cleaning
The data was cleaned to handle missing values, duplicates, and outliers.

Missing Values: The dataset was checked for any missing values, and it was found to be complete with no null entries.

Duplicates: There were 86,854 duplicated rows, which were removed to ensure the integrity of the model. After removing duplicates, the dataset had 96,146 unique samples.

Outliers: The data was checked for outliers in numerical columns like age, bmi, HbA1c_level, and blood_glucose_level. While some outliers were present, they were considered to be within a reasonable range for the given features.

Univariate Analysis
Univariate analysis was performed to understand the distribution of individual features.

Categorical Features: The distribution of categorical features like gender, hypertension, heart_disease, and smoking_history was analyzed. The gender feature had a small number of 'Other' entries, which were removed to maintain a binary classification for this feature.

Numerical Features: The distribution of numerical features was visualized using histograms to understand their spread and central tendency.

Multivariate Analysis
Multivariate analysis was conducted to explore the relationships between different features and their correlation with the target variable (diabetes).

Correlation Matrix: A correlation matrix was generated to visualize the relationships between numerical features. It was observed that HbA1c_level and blood_glucose_level have the highest correlation with diabetes, at 53% and 54%, respectively.

Data Preparation
Before training the models, the data was prepared through encoding and standardization.

Encoding Categorical Features
Categorical features were converted into a numerical format that can be used by the machine learning models.

Label Encoding: The gender feature was label encoded into numerical values.

One-Hot Encoding: The smoking_history feature was one-hot encoded to create separate binary columns for each category.

Train-Test Split
The dataset was split into training and testing sets, with 90% of the data used for training and 10% for testing.

Standardization
Numerical features were standardized using StandardScaler to ensure that they have a mean of 0 and a standard deviation of 1. This helps in improving the performance of the models.

Modeling
Three different machine learning models were trained and evaluated for this project:

K-Nearest Neighbors (KNN)

Random Forest

Boosting Algorithm (AdaBoost)

Each model was trained on the preprocessed training data.

Evaluation
The performance of the models was evaluated using two primary metrics: Mean Squared Error (MSE) and ROC-AUC score.

Mean Squared Error (MSE)
The MSE was calculated for both the training and testing sets to assess the models' performance. The Random Forest model achieved the lowest MSE on both the training and testing data, indicating its superior performance.

ROC-AUC Score
The ROC-AUC score was used to evaluate the models' ability to distinguish between positive and negative classes.

KNN ROC-AUC Score: 0.9853

Random Forest ROC-AUC Score: 0.9937

Boosting ROC-AUC Score: 0.9597

The Random Forest model had the highest ROC-AUC score, further confirming its effectiveness in this prediction task.

Prediction Test
A prediction test was conducted on a sample from the test set to see the models' predictions in action. All three models correctly predicted the outcome for the given sample.

How to Use
To use this project, you can follow these steps:

Clone the repository to your local machine.

Install the required libraries, including pandas, numpy, scikit-learn, and matplotlib.

Run the Jupyter Notebook (Diabetes_Prediction.ipynb) to see the entire analysis and model training process.

Use the trained models to make predictions on new data by following the data preparation steps outlined in the notebook.<img width="867" height="740" alt="Screenshot 2025-08-24 152225" src="https://github.com/user-attachments/assets/ac5b9ecc-8a17-40c1-b66a-b373819b4bca" />

