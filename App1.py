
import streamlit as st
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

model = joblib.load('decision_tree_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\n', ' ', text)
    return text

def predict_news(text):
    text = preprocess_text(text)
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)
    return 'Real News' if prediction == 1 else 'Fake News'

st.title("Fake News Detection App")
st.write("Enter A news Article Below To Check If It Is Classified As Real Or Fake.")
user_input = st.text_area("News Text", "Type Or Paste The News Article Here")


if st.button("Predict"):
    if user_input.strip():
        result = predict_news(user_input)
        st.write("The News Article Is Classified As:", result)
    else:
        st.write("Please Enter Some Text To Classify.")
