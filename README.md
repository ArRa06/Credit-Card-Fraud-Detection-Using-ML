# Credit Card Fraud Detection

A robust machine learning pipeline designed to detect fraudulent credit card transactions in highly imbalanced datasets. This project leverages optimized Random Forest and XGBoost classifiers, featuring automated data cleaning, feature scaling, and advanced evaluation metrics tailored for anomaly detection.

---

## 📊 The Dataset

This project uses the classic **Credit Card Fraud Detection** dataset.
* **Total Transactions:** ~284,807
* **Fraudulent Transactions:** ~492 (0.172%)
* **Features:** `Time`, `Amount`, and 28 PCA-transformed numerical features (`V1` - `V28`).

> **Note on Dataset Hosting:** > The uncompressed `creditcard.csv` file is approximately 150MB, which exceeds GitHub's 25MB file size limit. To run this project locally, please download the dataset directly from Kaggle using the link below and place the `creditcard.csv` file in the root directory.
> 
> 🔗 **[Download the Dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**

---

## 🚀 Key Features & Methodologies

### 1. Handling Extreme Class Imbalance
Standard machine learning models struggle when 99.8% of the data belongs to one class. This project avoids the "accuracy trap" by applying algorithmic-level balancing:
* **Random Forest:** Utilizes `class_weight="balanced"` to automatically adjust weights inversely proportional to class frequencies.
* **XGBoost:** Utilizes `scale_pos_weight` to apply a heavy penalty when the model fails to identify the minority class (Fraud).

### 2. Tailored Evaluation Metrics
Because the dataset is so heavily skewed, Accuracy is a misleading metric (a model predicting "Normal" 100% of the time would achieve 99.8% accuracy). This project evaluates models based on:
* **Recall:** The percentage of actual fraud cases successfully caught.
* **Precision:** The percentage of flagged transactions that were actually fraud (minimizing false positives).
* **AUPRC (Area Under the Precision-Recall Curve):** The definitive single metric for highly imbalanced binary classification.

### 3. Robust Data Preprocessing
* **Missing Value Handling:** Safely drops target variable `NaN` values to prevent scikit-learn pipeline crashes.
* **Feature Scaling:** Applies `StandardScaler` to `Time` and `Amount` features to ensure they do not statistically dominate the PCA features.
* **Stratified Splitting:** Uses `stratify=y` during train/test split to ensure the 0.17% fraud cases are evenly distributed between training and testing sets.

---

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost
* **Visualization:** Matplotlib, Seaborn

---

## 💻 Setup and Installation

**1. Download the project files:**
Download the repository files as a ZIP or to your local machine and open your terminal in the project directory.

**2. Install required dependencies:**
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
