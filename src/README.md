# 🧠 Source Code Overview

This folder contains the implementation of the phishing URL detection system.

## Files

* **phishing_detector.py**
  Core logic: feature extraction, model training, evaluation

* **phishing_lstm.py**
  Experimental deep learning models (LSTM, CNN)

* **run_basic.py**
  Main script (recommended) — fast execution

* **run_all.py**
  Full pipeline — runs all models (slow, experimental)

---

## ▶️ How to Run

Basic:

```
python run_basic.py
```

Advanced:

```
python run_all.py
```

---

## ⚠️ Notes

* Use `run_basic.py` for most use cases
* `run_all.py` is computationally heavy
* Dataset should be placed in root directory

---

## 🎯 Design Focus

* Modular structure
* Feature-based ML approach
* Performance vs efficiency trade-offs
