# MindCue: Mental Health Detection with Explainable AI & Carbon Tracking

## 📌 Project Overview
**MindCue** is an NLP-based Machine Learning system designed to analyze social media text and detect indicators of mental health challenges (e.g., Anxiety, Depression, Suicidal ideation).

Unlike standard classifiers, MindCue focuses on **transparency** and **sustainability**:
1.  **Explainable AI (XAI):** Uses **SHAP (SHapley Additive exPlanations)** to visualize exactly which words contributed to a specific prediction.
2.  **Green AI:** Integrates **CodeCarbon** to track and log the CO2 emissions generated during the model training process.

## 🚀 Key Features
- **Real-time Inference:** Flask API + Tailwind CSS frontend for instant text analysis.
- **Explainability:** Generates a probability map and highlights top contributing words using SHAP values.
- **Safety First:** Detects critical flags (e.g., "Suicidal") and provides immediate emergency resource prompts.
- **Eco-Conscious Training:** Tracks energy consumption (kWh) and carbon emissions (g) during model training.
- **Automated EDA:** Includes a script (`dataset_analyser.py`) to generate N-gram and label distribution plots automatically.

## 🛠️ Tech Stack
- **Backend:** Python (Flask), Joblib
- **ML & NLP:** Scikit-learn (Logistic Regression, TF-IDF), SHAP, NLTK
- **Frontend:** HTML5, Tailwind CSS, Vanilla JS
- **Sustainability:** CodeCarbon (Emissions Tracking)
- **Data Analysis:** Pandas, NumPy, Matplotlib, Seaborn

## 📂 Dataset
The model was trained on a combined dataset of social media posts (Twitter, Reddit) labeled with mental health statuses.

- **File Name:** `Combined Data.csv`
- **Columns:** `text` (User post), `label` (Status: *Anxiety, Normal, Depression, Suicidal*).

### 📥 How to Get the Data
Due to GitHub's file size limits, the dataset is hosted externally.
1. **Download:** [Sentiment Analysis for Mental Health (Kaggle)](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)
2. **Setup:** Move the downloaded `Combined Data.csv` file into the `data/` folder of this project.

*Note: The `train.py` script automatically looks for the file at `data/Combined Data.csv`.*

## 📂 Project Structure
```text
mindcue/
│
├── app.py               # Main Flask application (Inference + SHAP)
├── train.py             # ML Pipeline (Training + Carbon Tracking)
├── dataset_analyser.py  # Script for Exploratory Data Analysis (EDA)
├── parse_report.py      # Utility to parse carbon emission logs
├── requirements.txt     # Python dependencies
│
├── data/                # Folder for your dataset (Combined Data.csv)
├── templates/
│   └── index.html       # Single-page UI with Tailwind CSS
└── static/
    └── app.js           # Frontend logic for API calls
│
# ---------------------------------------------------------
# The following folders are generated automatically at runtime:
# ---------------------------------------------------------
├── models/              # (Generated) Stores saved model (.joblib)
└── logs/                # (Generated) Stores prediction logs
```
