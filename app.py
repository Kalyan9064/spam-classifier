import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------------- NLTK DOWNLOAD (FOR STREAMLIT CLOUD) ----------------
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="📧",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ---------------- TEXT PREPROCESS FUNCTION ----------------
def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)

    words = [w for w in words if w.isalnum()]
    words = [w for w in words if w not in stop_words]
    words = [ps.stem(w) for w in words]

    return " ".join(words)

# ---------------- UI ----------------
st.title("📧 AI Spam Classifier")
st.caption("Detect Spam Emails & SMS using Machine Learning")

st.markdown("---")

# Text input
input_sms = st.text_area(
    "✍️ Enter your message:",
    height=150,
    placeholder="Type your message here..."
)

# File upload option
uploaded_file = st.file_uploader("📄 Or upload a .txt file", type=["txt"])

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict"):

    # If file uploaded, read it
    if uploaded_file is not None:
        input_sms = uploaded_file.read().decode()

    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        with st.spinner("Analyzing..."):
            transformed = transform_text(input_sms)
            vector_input = tfidf.transform([transformed])

            prediction = model.predict(vector_input)[0]
            prob = model.predict_proba(vector_input)[0]

        confidence = max(prob) * 100

        st.markdown("---")

        if prediction == 1:
            st.error(f"🚨 SPAM DETECTED ({confidence:.2f}% confidence)")
        else:
            st.success(f"✅ NOT SPAM ({confidence:.2f}% confidence)")

# ---------------- SIDEBAR ----------------
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "This app uses NLP and a Machine Learning model "
    "to classify messages as Spam or Not Spam."
)

st.sidebar.markdown("### 🧠 Tech Stack")
st.sidebar.write("- Python")
st.sidebar.write("- Streamlit")
st.sidebar.write("- NLTK")
st.sidebar.write("- Scikit-learn")

st.sidebar.markdown("### 👨‍💻 Developer")
st.sidebar.write("Kalyan Chakraborty")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("© 2026 Spam Classifier | Built with ML & NLP")