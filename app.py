import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load and train the model once and cache it
@st.cache_resource
def load_model():
    # Read and clean data
    df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
    df.columns = ['label', 'text']
    df.dropna(subset=['label', 'text'], inplace=True)

    # Use string labels: 'ham' and 'spam'
    X = df['text']
    y = df['label']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('model', MultinomialNB())
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

# Main Streamlit UI
def main():
    st.set_page_config(page_title="Spam Detection", layout="centered")
    # Load external CSS file
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    st.title("📧 Email/SMS Spam Classifier")
    st.write("Enter a message and we'll tell you if it's spam or not!")

    user_input = st.text_area("✉️ Message", "")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter a message.")
        else:
            model = load_model()
            prediction = model.predict([user_input])[0]  # 'ham' or 'spam'
            result = "✅ Not Spam" if prediction == 'ham' else "🚫 Spam"
            st.success(f"Prediction: {result}")

if __name__ == '__main__':
    main()
