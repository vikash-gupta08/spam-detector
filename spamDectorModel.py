import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st
import requests
from io import StringIO


# 1. Load dataset
url = "https://drive.google.com/uc?id=1dMXbX9rrZsEIaJrME6EZYBnDl3uYpPrt"
response = requests.get(url)
data = pd.read_csv(StringIO(response.text))

# data = data[['v1', 'v2']]
data.columns = ['category', 'message']

# 2. Data cleaning
data.drop_duplicates(inplace=True)
data['category'] = data['category'].replace({'ham': 'Not Spam', 'spam': 'Spam'})

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

data['message'] = data['message'].apply(preprocess_text)

# 3. Splitting dataset
X = data['message']
y = data['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Vectorization
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Model training
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 6. Model evaluation
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 7. Predict function
def predict_message(msg):
    msg = preprocess_text(msg)
    msg_vec = vectorizer.transform([msg])
    prediction = model.predict(msg_vec)
    return prediction[0]

# 8. Simple Streamlit app UI for deployment
st.title("Spam Detection")

# user_input = st.text_area("Enter Message Here")
# if st.button("Predict"):
#     result = predict_message(user_input)
#     st.write(f"Prediction: {result}")


# Page configuration
st.set_page_config(page_title="Message Predictor", page_icon="💬", layout="centered")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
    }
    .stTextArea > label {
        font-size: 18px;
        font-weight: bold;
        color: #0d6efd;
    }
    .stButton button {
        background-color: #0d6efd;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border: none;
        border-radius: 6px;
        margin-top: 10px;
    }
    .stButton button:hover {
        background-color: #0b5ed7;
    }
    .prediction-box {
        padding: 15px;
        margin-top: 20px;
        border: 2px solid #0d6efd;
        border-radius: 8px;
        background-color: #e7f1ff;
        color: #0d6efd;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("💬 Message Prediction App")
st.markdown("Enter your message below and click **Predict** to get the result.")

# Input area with better label
user_input = st.text_area(
    "Enter your message here:",
    height=150,
    placeholder="Type something here..."
)

# Predict button and result display
if st.button("🔮 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message before predicting.")
    else:
        # Simulate prediction (replace with actual predict_message function)
        result = predict_message(user_input)  # Ensure this function is defined
        st.markdown(f"<div class='prediction-box'>Prediction: {result}</div>", unsafe_allow_html=True)
else:
    st.info("Enter a message and click **Predict** to see the result.")


# print(predict_message("HI"))
