# Week 5 — NLP & Sentiment Analysis Dashboard

Week 5 task for the AI/ML Pre-Internship Program at IT Simplera Institute (Roll No: AIMLB01-1730).

This project builds a sentiment classifier on real Amazon customer reviews and wraps it in an
interactive Streamlit dashboard where anyone can type a review and get an instant prediction.

## What's in here

- `notebooks/week5_nlp.ipynb` — full NLP pipeline: data loading, text cleaning, word clouds,
  TF-IDF vectorization, model training, evaluation, and model comparison
- `app.py` — Streamlit dashboard (Home / Data Overview / Sentiment Predictor)
- `model/` — saved model, vectorizer, and generated charts/word clouds
- `data/` — dataset used for training
- `requirements.txt` — Python dependencies

## Dataset

Amazon customer reviews with a review text column and a positive/negative sentiment label.
The label is imbalanced (most reviews are positive), which is handled in training by using
`class_weight="balanced"` rather than dropping data.

## Approach

1. **Cleaning:** lowercase text, strip HTML/URLs/punctuation, remove stopwords.
2. **Word clouds:** generated separately for positive and negative reviews to visualize the
   most common language in each class.
3. **Vectorization:** TF-IDF with unigrams and bigrams (5,000 features).
4. **Models:** Logistic Regression and Linear SVM, both trained with balanced class weights.
5. **Evaluation:** classification reports, confusion matrices, and a side-by-side accuracy/F1
   comparison chart. The better-performing model (by weighted F1) is saved along with the
   TF-IDF vectorizer for use in the dashboard.

## Running it locally

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

**Notebook:** open `notebooks/week5_nlp.ipynb` in VS Code, select the `venv` kernel, and run
all cells (the file paths inside assume it's run from the `notebooks/` folder, so the dataset
path `../data/amazon_reviews.csv` will resolve correctly).

**Dashboard:** run from the project root (not from inside `notebooks/`):

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints in your browser (usually `http://localhost:8501`).

## Dashboard pages

- **Home** — project introduction and a short explanation of sentiment analysis
- **Data Overview** — class distribution chart and the positive/negative word clouds
- **Sentiment Predictor** — type any review and get a predicted sentiment with a confidence score
