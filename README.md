# 🧠 Cardiovascular Risk Prediction (Decision Tree)

## 📌 Overview

This project uses a **Decision Tree model** to predict the probability of developing **cardiovascular disease (CVD)** within the next **10 years**.

The model is trained on the **Framingham Heart Study dataset**, a well-known medical dataset.

---

## ⚙️ Key Steps

### 1. Data Preprocessing

* Handled missing values
* Encoded categorical features
* Split data (80% training / 20% testing)

---

### 2. Handling Imbalanced Data

* Tested multiple techniques
* Final approach: **Downsampling** to balance classes

---

### 3. Model Optimization

* Used **GridSearchCV** for hyperparameter tuning
* Applied **Decision Tree pruning** to reduce overfitting

---

### 4. Feature Engineering

* Created new features (e.g., Pulse Pressure = `sysBP - diaBP`)
* Added interaction features
* Removed low-impact variables

---

### 5. Synthetic Data

* Used **CTGAN** to generate realistic patient data
* Tested model on synthetic samples

---

### 6. Model Saving

* Saved final model using **joblib** (`model.pkl`)

---

## 📊 Results

* Improved accuracy, precision, and recall
* Better detection of at-risk patients
* ~70% accuracy on final model

---

## 🛠 Technologies

* Python
* Pandas, NumPy
* Scikit-learn
* CTGAN
* Joblib

---

## ⚠️ Disclaimer

This project is for educational purposes only and not for medical use.
