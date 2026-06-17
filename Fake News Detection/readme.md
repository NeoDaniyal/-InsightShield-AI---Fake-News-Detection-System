# 📰 InsightShield AI - Fake News Detection System

## Overview

InsightShield AI is an end-to-end Machine Learning application designed to detect fake news headlines using Natural Language Processing (NLP) and Machine Learning techniques.

The project includes:

* Data Cleaning & Preprocessing
* Exploratory Data Analysis (EDA)
* TF-IDF Feature Engineering
* Multiple Machine Learning Models
* Deep Learning (LSTM)
* Interactive Streamlit Dashboard
* Plotly Analytics Visualizations
* Real-Time Fake News Prediction

---

## Features

### AI Prediction Portal

* Detects whether a news headline is Real or Fake
* Displays prediction confidence score
* Interactive confidence gauge meter
* Real-time inference using trained model

### Analytics Dashboard

* Real vs Fake News Distribution
* Dataset Statistics
* Tweet Spread Analysis
* Interactive Plotly Charts

### Model Performance Dashboard

* Logistic Regression Evaluation
* Random Forest Evaluation
* XGBoost Evaluation
* LSTM Evaluation
* Model Comparison Charts

---

## Dataset

Dataset Used:

**FakeNewsNet Dataset**

Dataset Size:

| Metric        | Value  |
| ------------- | ------ |
| Total Records | 23,196 |
| Real News     | 17,441 |
| Fake News     | 5,755  |

Columns:

* title
* news_url
* source_domain
* tweet_num
* real

---

## Tech Stack

### Data Science

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn

### Machine Learning

* Scikit-Learn
* XGBoost

### Deep Learning

* TensorFlow
* Keras
* LSTM Networks

### NLP

* NLTK
* TF-IDF Vectorization

### Dashboard

* Streamlit
* Plotly

---

## Machine Learning Pipeline

### Data Preprocessing

* Lowercase Conversion
* URL Removal
* Punctuation Removal
* Number Removal
* Stopword Removal
* Lemmatization

### Feature Engineering

TF-IDF Vectorization

* Maximum Features: 5000

### Models Trained

#### Logistic Regression

* Accuracy: 78%
* Weighted F1 Score: 0.75

#### Random Forest

* Accuracy: 79%
* Weighted F1 Score: 0.80

#### XGBoost

* Accuracy: 82%
* Weighted F1 Score: 0.81

#### LSTM Network

* Accuracy: 81%
* Weighted F1 Score: 0.81

---

## Best Model

🏆 **XGBoost Classifier**

Performance:

* Accuracy: 82%
* Weighted F1 Score: 0.81

XGBoost achieved the highest overall performance and was selected as the production model for deployment.

---

## Project Structure

```text
InsightShield-AI/
│
├── app.py
│
├── data/
│   └── FakeNewsNet.csv
│
├── models/
│   ├── xgb_model.pkl
│   ├── tfidf.pkl
│   └── tokenizer.pkl
│
├── pages/
│   ├── Analytics.py
│   ├── Dashboard.py
│   └── Model_Performance.py
│
├── assets/
│   └── logo.png
│
├── notebooks/
│   └── model_training.ipynb
│
├── requirements.txt
│
└── README.md
```

---

## Installation

Clone Repository

```bash
git clone https://github.com/your-username/InsightShield-AI.git
```

Move into Project Directory

```bash
cd InsightShield-AI
```

Install Dependencies

```bash
pip install -r requirements.txt
```

Run Application

```bash
streamlit run app.py
```

---

## Dashboard Preview

### Prediction Portal

* Enter a headline
* Analyze credibility
* View confidence score

### Analytics Dashboard

* Dataset Insights
* Interactive Visualizations
* Distribution Analysis

### Model Performance

* Model Comparison
* Evaluation Metrics
* Performance Charts

---

## Future Improvements

* BERT Transformer Integration
* Real-Time News API Integration
* Explainable AI (SHAP)
* User Authentication
* Multi-Language Fake News Detection
* Cloud Deployment
* Continuous Model Retraining

---

## Learning Outcomes

This project demonstrates practical skills in:

* Data Cleaning
* NLP
* Feature Engineering
* Machine Learning
* Deep Learning
* Model Evaluation
* Dashboard Development
* Data Visualization
* Model Deployment

---

## Author

Data Science & Machine Learning Project

Built using Python, Streamlit, Plotly, XGBoost, TensorFlow, and NLP techniques.
