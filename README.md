"""

# 🚖 NYC Taxi Trip Duration Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)  
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)](https://scikit-learn.org/stable/)  
[![Pandas](https://img.shields.io/badge/pandas-Data%20Analysis-yellowgreen)](https://pandas.pydata.org/)

---

## 📌 Overview

This repository contains a **machine learning pipeline** to predict **taxi trip durations in New York City**.  
The workflow includes **data preprocessing, feature engineering, model training, and evaluation**, with results persisted for reproducibility.

Key highlights:

- 🚀 **End-to-end ML pipeline** (from raw data to predictions).
- 🧮 **Feature engineering**: distances, directions, datetime features, seasonal bins.
- 🔧 **Data preprocessing**: log transforms, scaling, categorical encoding.
- 📈 **Models**: Ridge Regression + polynomial features (auto-selected degree).
- 📝 **Evaluation metrics**: RMSE & R².
- 💾 **Persistence**: trained models stored with metadata using `joblib`.

---

## 📂 Project Structure

```
NYC-Taxi-Trip-Duration/
│
├── data_sample/                 # Sample data for testing
├── full_data/                   # Complete dataset
├── models/                      # Saved models (.pkl)
├── saving_data/                 # Processed data storage
│
├── data_evaluation.py           # Model evaluation scripts
├── data_preparing.py            # Data preparation and feature engineering
├── data_preprocessing.py        # Data cleaning and preprocessing
├── data_training.py             # Model training pipeline
├── feature_fragmenting.py       # Feature selection/engineering utilities
├── loading_model.py             # Model loading utilities
├── loading_test.py              # Model testing and validation
├── requirements.txt             # Dependencies
└── README.md                    # Documentation
```

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/AhmedYasser06/NYC-Taxi-Trip-Duration.git
cd NYC-Taxi-Trip-Duration-main
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1️⃣ Train the Model

```bash
python train.py
```

- Trains Ridge regression with polynomial features (degree auto-selected).
- Saves trained model + preprocessors + metadata into `models/`.

### 2️⃣ Evaluate on Validation / Test Sets

```bash
python load_test.py
```

- Loads saved model.
- Prepares data (feature engineering, preprocessing).
- Evaluates RMSE & R².

---

## 📊 Example Results

```
Train RMSE = 0.4373 | R² = 0.6972
Val   RMSE = 0.4424 | R² = 0.6942
Test  RMSE = 0.4451 | R² = 0.6920
```

---

## 📦 Dependencies

Main libraries:

- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [joblib](https://joblib.readthedocs.io/)

(see `requirements.txt` for the full list)
