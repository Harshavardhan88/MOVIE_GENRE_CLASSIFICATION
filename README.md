# 🎬 Movie Genre Classification using NLP and Machine Learning

This project focuses on building a **multi-label movie genre classification system** using **Natural Language Processing (NLP)** and **Machine Learning** techniques.  
Given a movie plot summary or description, the model predicts one or more suitable genres (e.g., Action, Drama, Comedy, etc.).

---

## 🚀 Overview

With the vast amount of movie data available online, automated genre tagging helps in organizing, recommending, and analyzing films.  
This project applies **text preprocessing, TF-IDF vectorization, and classical ML algorithms** to predict genres based on textual content.

---

## 🧠 Features

- Multi-label text classification (each movie can have multiple genres)
- Text preprocessing and feature extraction using **TF-IDF**
- Model training and comparison using:
  - **Logistic Regression**
  - **Random Forest Classifier**
- Evaluation using **Precision**, **Recall**, and **F1-score**
- Visualization of metrics and confusion matrices
- Model versioning and saving using **joblib**

---

## 🧩 Tech Stack

**Languages & Tools:**  
Python, scikit-learn, NLTK, Pandas, NumPy, Matplotlib, Seaborn

**Core Concepts:**  
Text Preprocessing, NLP, Multi-Label Classification, TF-IDF, Model Evaluation

---

## ⚙️ Workflow

1. **Data Collection**  
   Import dataset containing movie titles, descriptions, and corresponding genres.

2. **Data Preprocessing**  
   - Text cleaning (lowercasing, punctuation removal)
   - Tokenization and Lemmatization  
   - Stopword removal using NLTK

3. **Feature Extraction**  
   Convert text to numerical vectors using **TF-IDF Vectorizer**.

4. **Model Building**  
   Train models using Logistic Regression and Random Forest.

5. **Evaluation**  
   - Compute **F1-score**, **precision**, and **recall** for each genre.
   - Plot performance metrics.

6. **Model Saving**  
   Save trained models using joblib for future inference.

---

## 📊 Results

| Model               | Precision | Recall | F1-Score |
|----------------------|-----------|--------|----------|
| Logistic Regression  | 0.83      | 0.79   | 0.81     |
| Random Forest        | 0.78      | 0.74   | 0.76     |

*(Results may vary based on dataset size and preprocessing parameters.)*

---

## 🧪 Future Enhancements

- Integrate **deep learning models** (BERT / DistilBERT)
- Add **MLOps workflow** for experiment tracking with MLflow
- Deploy as an **API** using Flask or FastAPI
- Extend to **recommendation systems** using content similarity

---

## 📁 Project Structure

📦 MOVIE_GENRE_CLASSIFICATION
┣ 📜 movie_genre_classification.ipynb
┣ 📜 dataset.csv
┣ 📜 requirements.txt
┣ 📜 model_logistic.pkl
┣ 📜 model_randomforest.pkl
┗ 📜 README.md


