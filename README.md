# 📰 Fake News Detection System Using NLP

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📌 Project Overview

Fake news has become a major challenge in the digital world. This project uses **Natural Language Processing (NLP)** and **Machine Learning** to classify news articles as **Real** or **Fake**.

The application preprocesses news articles, converts them into numerical vectors using **TF-IDF Vectorization**, and predicts whether the article is fake or real using a **Logistic Regression** model.

An interactive **Streamlit** web application allows users to test news articles in real time.


# 🚀 Features

- ✅ Fake News Detection
- ✅ Real News Detection
- ✅ Confidence Score
- ✅ Fake News Probability
- ✅ Real News Probability
- ✅ Interactive Streamlit Web Application
- ✅ Machine Learning Model
- ✅ Text Preprocessing using NLTK
- ✅ TF-IDF Feature Extraction


# 🛠️ Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Programming Language |
| Pandas | Data Manipulation |
| NumPy | Numerical Computing |
| NLTK | Text Cleaning |
| Scikit-Learn | Machine Learning |
| TF-IDF | Feature Extraction |
| Logistic Regression | Classification Model |
| Joblib | Model Saving |
| Streamlit | Web Application |

---

# 📂 Project Structure

```
Fake-News-Detection/
│
├── Data/
│   ├── Fake.csv
│   └── True.csv
│
├── model/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebook/
│   └── FakeNewsDetection.ipynb
│
├── screenshots/
│   ├── home.png
│   ├── prediction_real.png
│   └── prediction_fake.png
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

# 📊 Dataset

The model is trained using the **Fake and Real News Dataset** available on Kaggle.

Dataset Link:

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Dataset contains

- Fake News Articles
- Real News Articles

---

# ⚙️ Machine Learning Pipeline

```
Collect Dataset
        │
        ▼
Load Dataset
        │
        ▼
Text Cleaning
        │
        ▼
Remove Stop Words
        │
        ▼
TF-IDF Vectorization
        │
        ▼
Train-Test Split
        │
        ▼
Logistic Regression
        │
        ▼
Model Evaluation
        │
        ▼
Save Model (.pkl)
        │
        ▼
Streamlit Deployment
```

---

# 🧹 Text Preprocessing

The following preprocessing steps are applied:

- Convert text to lowercase
- Remove URLs
- Remove HTML tags
- Remove punctuation
- Remove numbers
- Remove stop words
- Remove unnecessary spaces

---

# 🤖 Machine Learning Model

Algorithm Used

- Logistic Regression

Feature Extraction

- TF-IDF Vectorizer

---

# 📈 Evaluation Metrics

The model is evaluated using

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---


# 📦 Model Files

The trained model is stored using Joblib.

```
fake_news_model.pkl
```

```
tfidf_vectorizer.pkl
```

These files are loaded directly into the Streamlit application for prediction.

---

# 🎯 Future Improvements

- Deep Learning (LSTM)
- BERT
- RoBERTa
- DistilBERT
- Explainable AI (LIME)
- SHAP Visualizations
- Cloud Deployment
- REST API
- Multi-language News Detection
- News Source Verification

---

# 🎓 Learning Outcomes

Through this project, I learned

- Natural Language Processing
- Text Cleaning
- TF-IDF Vectorization
- Logistic Regression
- Machine Learning Workflow
- Model Evaluation
- Streamlit Deployment
- Model Serialization using Joblib

---

# 📄 Requirements

```
streamlit
pandas
numpy
scikit-learn
nltk
joblib
```

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Kanahi kumar**

B.Tech Computer Science Engineering (AI & ML)

Maharishi Markandeshwar (Deemed to be University)

GitHub: https://github.com/kanhairazz/kanhai

LinkedIn: https://linkedin.com/in/www.linkedin.com/in/kanhai-kumar-1b94b3320

---

# ⭐ If you found this project helpful, don't forget to star this repository!
