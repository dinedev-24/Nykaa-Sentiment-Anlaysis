import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt
import seab as sns

# Load BERT model and tokenizer
MODEL_PATH = "bert-base-uncased"
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

# Streamlit UI
def main():
    st.title("Nykaa Product Sentiment Analysis")
    st.write("Analyze product reviews using BERT-powered sentiment analysis!")

    # User Input Section
    st.subheader("Enter a Review for Sentiment Prediction")
    user_input = st.text_area("Review Text", "Type a review here...")
    if st.button("Analyze Sentiment"):
        sentiment = predict_sentiment(user_input)
        st.success(f"Predicted Sentiment: {sentiment}")
    
    # Upload File for Batch Processing
    st.subheader("Upload a CSV File for Bulk Sentiment Analysis")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "review_text" in df.columns:
            df["predicted_sentiment"] = df["review_text"].apply(predict_sentiment)
            st.write("Processed Data:")
            st.dataframe(df.head())
        else:
            st.error("CSV must contain a column named 'review_text'.")
    
    # Display Aspect-Based Sentiment Analysis Results
    st.subheader("Aspect-Based Sentiment Insights")
    aspect_data = {"Price": 1650, "Quality": 1600, "Fragrance": 850, "Packaging": 450}
    aspect_df = pd.DataFrame(list(aspect_data.items()), columns=["Aspect", "Mentions"])
    fig, ax = plt.subplots()
    sns.barplot(x="Aspect", y="Mentions", data=aspect_df, ax=ax)
    st.pyplot(fig)
    
if __name__ == "__main__":
    main()
