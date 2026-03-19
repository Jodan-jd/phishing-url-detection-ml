# 🤖 Phishing URL Detection using Machine Learning & Deep Learning

## 📌 Overview

This project explores phishing URL detection using both traditional machine learning and deep learning approaches. Instead of relying on static blacklists, the system identifies malicious URLs through structural and statistical patterns extracted from URLs.

---

## 🚀 Key Results

* **Best Model:** Feedforward Neural Network
* **F1 Score:** 99.21%
* **Precision:** 99.86%
* **Recall:** 98.56%
* **Training Time:** ~15 minutes (CPU)

---

## 🧠 Approach

### 🔍 Feature Engineering

* Extracted **24 handcrafted features** from URLs
* Focus areas:

  * URL structure
  * Character distribution
  * Security indicators
  * Domain-based patterns

### 🤖 Models Implemented

* Feedforward Neural Network (Best performing)
* Deep Neural Network
* Support Vector Machine (SGD)
* Logistic Regression

### 🧪 Advanced Models (Optional)

* LSTM
* CNN
* Hybrid models

👉 **Key Insight:**
Sequential models like LSTM significantly underperformed (~68% F1), indicating that phishing detection benefits more from structured feature engineering than sequence modeling.

---

## 📊 Dataset

* Source: Mendeley Dataset
* Size: 336,749 URLs
* Balanced dataset (phishing vs legitimate)

📥 Download dataset:
https://data.mendeley.com/datasets/m2479kmybx/1

Place dataset in the root directory as:

```
phishing_dataset.csv
```

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* TensorFlow
* Pandas / NumPy
* Matplotlib / Seaborn

---

## ⚙️ Installation

```
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 🔹 Basic Execution (Recommended)

```
python src/run_basic.py
```

✔️ This will:

* Automatically detect dataset format
* Train models
* Generate results and visualizations

---

### 🔹 Full Experiment Pipeline (Optional)

```
python src/run_all.py
```

⚠️ Notes:

* Runs all models including LSTM/CNN
* Training time can exceed **10+ hours**
* Not optimized for all systems

---

## 📂 Project Structure

```
phishing-url-detection-ml/
├── docs/
│   └── project-report.pdf
├── src/
│   ├── phishing_detector.py
│   ├── phishing_lstm.py
│   ├── run_basic.py
│   └── run_all.py
└── requirements.txt
```

---

## 📄 Project Report

👉 View Full Report:
[docs/project-report.pdf](https://github.com/Jodan-jd/phishing-url-detection-ml/tree/main/docs)

---

## 🎯 Key Learnings

* Feature engineering is critical in cybersecurity ML problems
* Simpler models can outperform complex deep learning approaches
* Real-world systems require balancing accuracy with efficiency
* Not all problems benefit from sequence-based models

---

## 🧠 Perspective

This project reflects my approach toward:

* Security-focused machine learning
* System-level thinking over tool-based implementation
* Understanding trade-offs between model complexity and performance

---

## ⚠️ Disclaimer

* Advanced models (LSTM/CNN) are experimental
* Results may vary depending on system configuration
* For most use cases, `run_basic.py` is recommended

---


