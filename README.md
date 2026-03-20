# 🩺 <span style="color:#00C4FF; font-size:40px;">DiaPredict AI</span>  
### 📊 Intelligent Diabetes Risk Prediction System  

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Project-Active-brightgreen?style=for-the-badge)

---

> A machine learning system designed to **predict diabetes risk using patient health data**, enabling early detection and better clinical decision-making.

---

## 🔍 Overview

Early detection of diabetes can significantly reduce long-term health complications.  

**DiaPredict AI** leverages multiple machine learning models to analyze patient attributes such as **age, BMI, glucose levels, and lifestyle factors**, delivering accurate predictions of diabetes likelihood.

---

## ✨ Key Features

- 🧠 **Multi-Model Approach**  
  Utilizes KNN, Random Forest, and Boosting algorithms for robust predictions  

- 📊 **Comprehensive Data Analysis**  
  Includes univariate and multivariate analysis for deeper insights  

- ⚙️ **Data Preprocessing Pipeline**  
  Handles encoding, scaling, and data cleaning efficiently  

- 📈 **High Model Performance**  
  Achieves strong ROC-AUC scores across models  

- 🔍 **Feature Impact Analysis**  
  Identifies key factors influencing diabetes risk  

---

## 🛠️ Tech Stack

| Category        | Tools Used |
|----------------|-----------|
| Language       | Python 3.x |
| ML Libraries   | Scikit-learn |
| Data Handling  | Pandas, NumPy |
| Visualization  | Matplotlib |

---

## 🖥️ Workflow

- 📂 **Data Ingestion**  
  Processes a dataset of **180K+ patient records** with health indicators  

- 🧹 **Data Cleaning & Validation**  
  Removes duplicates (~86K rows), checks missing values, and handles inconsistencies  

- 📊 **Exploratory Data Analysis**  
  - Distribution analysis of features  
  - Correlation study with diabetes outcome  

- ⚙️ **Feature Engineering**  
  - Label Encoding (gender)  
  - One-Hot Encoding (smoking history)  
  - Standardization using `StandardScaler`  

- 🔀 **Train-Test Split**  
  90% training and 10% testing for reliable evaluation  

---

## 🤖 Modeling

- 🔍 **K-Nearest Neighbors (KNN)**  
  Captures local patterns in patient data  

- 🌲 **Random Forest (Best Performer)**  
  Provides high accuracy and robustness  

- ⚡ **Boosting (AdaBoost)**  
  Improves weak learners for better predictions  

---

## 📈 Model Performance

- 🥇 **Random Forest**  
  - ROC-AUC: **0.9937** (Best)  
  - Lowest Mean Squared Error  

- 🥈 **KNN**  
  - ROC-AUC: **0.9853**  

- 🥉 **AdaBoost**  
  - ROC-AUC: **0.9597**  

✔️ Random Forest emerged as the most effective model for this task  

---

## 📊 Key Insights

- 📌 **Blood Glucose Level** and **HbA1c Level** are the strongest predictors  
- 📈 Both features show **~53–54% correlation** with diabetes  
- 🧹 Data quality significantly improved after removing duplicates  

---

## 📊 Notebook & Implementation

- 📓 `Diabetes_Prediction.ipynb`  
  Contains full pipeline:
  - Data Cleaning  
  - EDA  
  - Feature Engineering  
  - Model Training  
  - Evaluation  

---

## 🚀 Usage

- 📥 Clone the repository  
- ⚙️ Install dependencies:
  ```bash
  pip install pandas numpy scikit-learn matplotlib
 ```
jupyter notebook Diabetes_Prediction.ipynb
```
##🔍 Follow the pipeline to:

-Train models

-Evaluate performance

-Predict on new patient data


