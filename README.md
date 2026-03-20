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
<img width="889" height="839" alt="Screenshot 2025-08-24 152421" src="https://github.com/user-attachments/assets/5ff8914c-09fd-42bf-85d5-fbcafd3e59b1" />
<img width="510" height="359" alt="Screenshot 2025-08-24 152430" src="https://github.com/user-attachments/assets/0128c8a5-2c82-4e20-9501-34dde32a1c56" />
<img width="1277" height="652" alt="Screenshot 2025-08-24 152409" src="https://github.com/user-attachments/assets/f6ab8b5a-2006-4281-8784-bc63730de938" />
<img width="1171" height="876" alt="Screenshot 2025-08-24 152315" src="https://github.com/user-attachments/assets/38e50471-8575-481f-98fb-ec1f72c0f619" />

<img width="1288" height="649" alt="Screenshot 2025-08-24 152401" src="https://github.com/user-attachments/assets/9de0a1a4-05bb-4877-b56c-594a871de609" />

<img width="867" height="740" alt="Screenshot 2025-08-24 152225" src="https://github.com/user-attachments/assets/ac5b9ecc-8a17-40c1-b66a-b373819b4bca" />

