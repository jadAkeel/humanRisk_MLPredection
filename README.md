# 🧠 Cardiovascular Risk Prediction (Decision Tree)

## 📌 Overview

This project uses a **Decision Tree model** to predict the probability of developing **cardiovascular disease (CVD)** within the next **10 years**.

The model is trained on the **Framingham Heart Study dataset**, a well-known medical dataset.

---

## ⚙️ Key Steps

### 1. Data Preprocessing

* Handled missing values
* Encoded categorical features
* Split data (70% training / 30% held-out testing), stratified by outcome

---

### 2. Handling Imbalanced Data

* Tested multiple techniques
* Final approach: **Downsampling** to balance classes

---

### 3. Model Optimization

* Used **GridSearchCV** for hyperparameter tuning
* Selected the **Decision Tree pruning** parameter with 5-fold cross-validation on the training set

---

### 4. Feature Engineering

* Created new features (e.g., Pulse Pressure = `sysBP - diaBP`)
* Added interaction features
* Removed low-impact variables
* Learned clipping and normalization statistics from the training partition only to keep the held-out evaluation isolated

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
* Approximately 70% accuracy on the held-out test set

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

The included project report has student identifiers removed. Synthetic patient
records are generated for technical evaluation and are not real patient data.
