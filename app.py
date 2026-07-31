import re
import joblib
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

st.set_page_config(page_title="Sentiment Analysis Dashboard", page_icon="💬", layout="wide")

# ---------------------------------------------------------------------------
# Text cleaning — must match the cleaning used in the training notebook
# ---------------------------------------------------------------------------
stop_words = set(ENGLISH_STOP_WORDS)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 1]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("model/sentiment_model.pkl")
    vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
    return model, vectorizer


@st.cache_data
def load_dataset():
    df = pd.read_csv("data/amazon_reviews.csv")
    df = df.dropna(subset=["verified_reviews"]).reset_index(drop=True)
    df = df.rename(columns={"verified_reviews": "review", "feedback": "sentiment"})
    df["sentiment_label"] = df["sentiment"].map({1: "positive", 0: "negative"})
    return df


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Overview", "Sentiment Predictor"])

# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------
if page == "Home":
    st.title("Sentiment Analysis Dashboard")
    st.markdown(
        """
        This project applies **Natural Language Processing (NLP)** to real Amazon customer
        reviews to predict whether a review expresses **positive** or **negative** sentiment.

        ### What is Sentiment Analysis?
        Sentiment analysis is the process of using NLP to automatically determine the
        emotional tone behind a piece of text — commonly classified as positive, negative,
        or neutral. It's widely used to analyze customer feedback, monitor brand reputation,
        and power recommendation and support systems at scale.

        ### About this project
        - **Dataset:** Amazon customer reviews with review text and a positive/negative label
        - **Cleaning:** lowercasing, noise/URL removal, stopword removal
        - **Vectorization:** TF-IDF (unigrams + bigrams)
        - **Models trained:** Logistic Regression and Linear SVM
        - **Deployed model:** the better-performing model of the two, selected by weighted F1 score

        Use the sidebar to explore the dataset (**Data Overview**) or try the model yourself
        on any review text (**Sentiment Predictor**).
        """
    )

# ---------------------------------------------------------------------------
# Data Overview
# ---------------------------------------------------------------------------
elif page == "Data Overview":
    st.title("Data Overview")

    df = load_dataset()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Class Distribution")
        counts = df["sentiment_label"].value_counts()
        st.bar_chart(counts)
        st.dataframe(counts.rename("count"))

    with col2:
        st.subheader("Dataset Snapshot")
        st.write(f"Total reviews: **{len(df)}**")
        st.dataframe(df[["review", "sentiment_label"]].head(10))

    st.subheader("Word Clouds")
    wc_col1, wc_col2 = st.columns(2)
    with wc_col1:
        st.markdown("**Positive Reviews**")
        st.image("model/wordcloud_positive.png", use_container_width=True)
    with wc_col2:
        st.markdown("**Negative Reviews**")
        st.image("model/wordcloud_negative.png", use_container_width=True)

# ---------------------------------------------------------------------------
# Sentiment Predictor
# ---------------------------------------------------------------------------
elif page == "Sentiment Predictor":
    st.title("Sentiment Predictor")
    st.markdown("Type or paste a product review below and click **Predict** to see its sentiment.")

    model, vectorizer = load_model_and_vectorizer()

    user_review = st.text_area("Review text", height=150, placeholder="e.g. This product works great and arrived fast!")

    if st.button("Predict"):
        if not user_review.strip():
            st.warning("Please enter a review first.")
        else:
            cleaned = clean_text(user_review)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]
            confidence = proba[pred]

            label = "Positive 🙂" if pred == 1 else "Negative 🙁"
            st.subheader(f"Predicted Sentiment: {label}")
            st.write(f"Confidence: **{confidence * 100:.2f}%**")

            st.progress(float(confidence))
