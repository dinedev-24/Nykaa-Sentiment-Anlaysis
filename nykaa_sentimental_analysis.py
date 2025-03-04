import os
import subprocess
import streamlit as st
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from transformers import BertTokenizer, BertForSequenceClassification

import subprocess

# Ensure pip, setuptools, and wheel are updated
subprocess.run(["pip", "install", "--upgrade", "pip", "setuptools", "wheel"], check=True)

# Install tokenizers **without building from source**
subprocess.run(["pip", "install", "--prefer-binary", "tokenizers"], check=True)

# Install the rest of the dependencies
subprocess.run(["pip", "install", "-r", "requirements.txt", "--no-cache-dir"], check=True)


# ✅ Set Streamlit page config
st.set_page_config(page_title="Nykaa Sentiment Analysis", layout="wide")

# ✅ Load BERT Model & Tokenizer (Handle Missing Files)
MODEL_PATH = "./model"  # Make sure you upload this folder in your repo

try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=3)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()  # Stop execution if model fails to load

# ✅ Define Sentiment Labels
label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}

# ✅ Function to Predict Sentiment
def predict_sentiment(review_text):
    inputs = tokenizer(review_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return label_mapping[prediction]

# ✅ Function to Generate a Word Cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    return fig

# ✅ Load Cleaned Dataset
@st.cache_data
def load_data():
    file_path = "nykaa_cleaned_reviews.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.warning("⚠️ Dataset not found! Please upload 'nykaa_cleaned_reviews.csv'.")
        return pd.DataFrame()

# ✅ Main Streamlit App
def main():
    st.title("🛍️ Nykaa Product Sentiment Analysis")
    st.write("Analyze customer reviews using AI-powered sentiment analysis!")

    # 🎯 Sidebar Navigation
    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Select an option:", ["Home", "Visualizations", "Bulk Analysis"])

    # 🏠 Home - Single Review Analysis
    if option == "Home":
        st.subheader("Enter a Review for Sentiment Prediction")
        user_input = st.text_area("Review Text", "Type a review here...")
        if st.button("Analyze Sentiment"):
            sentiment = predict_sentiment(user_input)
            st.success(f"Predicted Sentiment: {sentiment}")

    # 📊 Visualizations
    elif option == "Visualizations":
        df = load_data()
        if df.empty:
            return  # Stop execution if no data

        # 🎯 Sentiment Distribution
        st.subheader("Sentiment Distribution")
        sentiment_counts = df["sentiment"].value_counts()
        fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title="Sentiment Distribution")
        st.plotly_chart(fig)

        # 🎯 Aspect-Based Sentiment Breakdown
        st.subheader("Aspect-Based Sentiment Breakdown")
        aspect_data = {"Price": 1650, "Quality": 1600, "Fragrance": 850, "Packaging": 450}
        aspect_df = pd.DataFrame(list(aspect_data.items()), columns=["Aspect", "Mentions"])
        fig, ax = plt.subplots()
        sns.barplot(x="Aspect", y="Mentions", data=aspect_df, ax=ax)
        st.pyplot(fig)

        # 🎯 Word Cloud
        st.subheader("Word Cloud of Most Common Words in Reviews")
        text = " ".join(df["review_text"].dropna().astype(str))  # Handle NaN values
        st.pyplot(generate_wordcloud(text))

        # 🎯 Sentiment Trend Over Time
        st.subheader("Sentiment Trend Over Time")
        df["review_date"] = pd.to_datetime(df["review_date"])
        sentiment_trend = df.groupby(df["review_date"].dt.to_period("M")).size()
        fig = px.line(x=sentiment_trend.index.astype(str), y=sentiment_trend.values, title="Sentiment Trend Over Time")
        st.plotly_chart(fig)

    # 📂 Bulk Analysis - CSV Upload
    elif option == "Bulk Analysis":
        st.subheader("Upload a CSV File for Bulk Sentiment Analysis")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            # 🔍 Ensure "review_text" column exists
            if "review_text" not in df.columns:
                st.error("CSV must contain a column named 'review_text'. Please check the uploaded file.")
                return  # Stop execution

            # 🔍 Predict Sentiment for Each Review
            df["predicted_sentiment"] = df["review_text"].apply(predict_sentiment)
            st.write("Processed Data:")
            st.dataframe(df.head())

if __name__ == "__main__":
    main()
