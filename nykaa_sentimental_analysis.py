import os
import subprocess

# Ensure pip is up to date
subprocess.run(["pip", "install", "--upgrade", "pip"], check=True)

# Install missing dependencies
dependencies = [
    "setuptools>=75.0.0",
    "wheel",
    "numpy>=1.26.0",
    "pandas==2.2.3",
    "streamlit==1.25.0",
    "torch==2.2.0",
    "transformers==4.31.0",  # Let transformers decide the tokenizers version
    "tokenizers>=0.12.0,<0.14",  # Install a compatible tokenizers version
    "seaborn==0.13.2",
    "matplotlib==3.10.1",
    "plotly==6.0.0",
    "wordcloud==1.9.4",
    "python-dateutil>=2.8.2",
    "pytz>=2020.1",
    "tzdata>=2022.7",
    "altair<6,>=4.0",
    "blinker<2,>=1.0.0",
    "cachetools<6,>=4.0",
    "click<9,>=7.0",
    "packaging<24,>=16.8",
    "pillow==9.5.0",
    "protobuf<5,>=3.20",
    "pyarrow>=6.0",
    "pympler<2,>=0.9",
    "requests<3,>=2.18",
    "rich<14,>=10.14.0",
    "tenacity<9,>=8.1.0",
    "toml<2,>=0.10.1",
    "typing-extensions<5,>=4.1.0",
    "tzlocal<5,>=1.1",
    "validators<1,>=0.2",
    "gitpython!=3.1.19,<4,>=3.0.7",
    "pydeck<1,>=0.8",
    "tornado<7,>=6.0.3",
    "watchdog>=2.1.5",
    "filelock",
    "sympy",
    "networkx",
    "jinja2",
    "fsspec",
    "huggingface-hub<1.0,>=0.11.0",
    "pyyaml>=5.1",
    "regex!=2019.12.17",
    "tqdm>=4.27"
]

# Install dependencies dynamically
for package in dependencies:
    try:
        subprocess.run(["pip", "install", package, "--no-cache-dir"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")

# Import after installation
import streamlit as st
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from transformers import BertTokenizer, BertForSequenceClassification

# Load BERT model and tokenizer
MODEL_PATH = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=3)


import streamlit as st
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from transformers import BertTokenizer, BertForSequenceClassification

# Set page config before UI elements
st.set_page_config(page_title="Nykaa Sentiment Analysis", layout="wide")

# Load BERT model and tokenizer (Use local files)
MODEL_PATH = "./model"  # Make sure you upload the model folder in your repo
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=3)

# Define sentiment labels
label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Function to predict sentiment
def predict_sentiment(review_text):
    inputs = tokenizer(review_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return label_mapping[prediction]

# Function to generate a word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    return fig

# Load cleaned dataset
@st.cache_data
def load_data():
    return pd.read_csv("nykaa_cleaned_reviews.csv")

def main():
    st.title("🛍️ Nykaa Product Sentiment Analysis")
    st.write("Analyze customer reviews using AI-powered sentiment analysis!")
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Select an option:", ["Home", "Visualizations", "Bulk Analysis"])
    
    if option == "Home":
        st.subheader("Enter a Review for Sentiment Prediction")
        user_input = st.text_area("Review Text", "Type a review here...")
        if st.button("Analyze Sentiment"):
            sentiment = predict_sentiment(user_input)
            st.success(f"Predicted Sentiment: {sentiment}")
    
    elif option == "Visualizations":
        df = load_data()
        
        st.subheader("Sentiment Distribution")
        sentiment_counts = df["sentiment"].value_counts()
        fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title="Sentiment Distribution")
        st.plotly_chart(fig)
        
        st.subheader("Aspect-Based Sentiment Breakdown")
        aspect_data = {"Price": 1650, "Quality": 1600, "Fragrance": 850, "Packaging": 450}
        aspect_df = pd.DataFrame(list(aspect_data.items()), columns=["Aspect", "Mentions"])
        fig, ax = plt.subplots()
        sns.barplot(x="Aspect", y="Mentions", data=aspect_df, ax=ax)
        st.pyplot(fig)
        
        st.subheader("Word Cloud of Most Common Words in Reviews")
        text = " ".join(df["review_text"].dropna().astype(str))  # Fix NaN values
        st.pyplot(generate_wordcloud(text))
        
        st.subheader("Sentiment Trend Over Time")
        df["review_date"] = pd.to_datetime(df["review_date"])
        sentiment_trend = df.groupby(df["review_date"].dt.to_period("M")).size()
        fig = px.line(x=sentiment_trend.index.astype(str), y=sentiment_trend.values, title="Sentiment Trend Over Time")
        st.plotly_chart(fig)
    
    elif option == "Bulk Analysis":
        st.subheader("Upload a CSV File for Bulk Sentiment Analysis")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            # Ensure "review_text" column exists
            if "review_text" not in df.columns:
                st.error("CSV must contain a column named 'review_text'. Please check the uploaded file.")
                return  # Stop execution
            
            df["predicted_sentiment"] = df["review_text"].apply(predict_sentiment)
            st.write("Processed Data:")
            st.dataframe(df.head())

if __name__ == "__main__":
    main()
