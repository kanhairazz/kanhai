import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Title
st.title("📰 Fake News Detection System")
st.write("Enter a news article below to check whether it is Real or Fake.")

# Input
news = st.text_area("Enter News Article")

# Button
if st.button("Predict"):

    if news.strip() == "":
        st.warning("Please enter a news article.")

    else:

        # Convert text
        vector = vectorizer.transform([news])

        # Prediction
        prediction = model.predict(vector)

        # Probability
        probability = model.predict_proba(vector)

        confidence = max(probability[0]) * 100

        fake_prob = probability[0][0] * 100
        real_prob = probability[0][1] * 100

        # Output
        if prediction[0] == 1:
            st.success("✅ Real News")
        else:
            st.error("❌ Fake News")

        st.write(f"Confidence: {confidence:.2f}%")
        st.write(f"Fake News Probability: {fake_prob:.2f}%")
        st.write(f"Real News Probability: {real_prob:.2f}%")
