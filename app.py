import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="🛡️",
    layout="centered"
)

# ---------------- GLASSMORPHISM CSS ----------------
st.markdown("""
<style>/home/kalyan-chakraborty/Downloads/pycharm-2025.3.2.1

body {
    background: linear-gradient(135deg,#0f172a,#1e293b);
    color:white;
}

.big-title {
    text-align:center;
    font-size:42px;
    font-weight:800;
    background: linear-gradient(90deg,#38bdf8,#6366f1);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.subtitle {
    text-align:center;
    color:#cbd5f5;
    margin-bottom:25px;
}

.card {
    padding:25px;
    border-radius:15px;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border:1px solid rgba(255,255,255,0.1);
}

.result-spam {
    padding:20px;
    border-radius:12px;
    background:#7f1d1d;
    text-align:center;
    font-size:22px;
    font-weight:700;
}

.result-safe {
    padding:20px;
    border-radius:12px;
    background:#064e3b;
    text-align:center;
    font-size:22px;
    font-weight:700;
}

.stButton>button {
    background: linear-gradient(90deg,#6366f1,#22d3ee);
    color:white;
    border:none;
    border-radius:10px;
    font-size:18px;
    font-weight:600;
    padding:12px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    stop_words = set(stopwords.words('english'))

    words = [
        ps.stem(w)
        for w in text
        if w.isalnum() and w not in stop_words
    ]

    return " ".join(words)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# ---------------- HEADER ----------------
st.markdown("<div class='big-title'>🛡️ AI Spam Detector</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Smart Email & SMS Protection using Machine Learning</div>", unsafe_allow_html=True)

# ---------------- INPUT CARD ----------------

input_sms = st.text_area(
    "✍️ Enter your message",
    height=140,
    placeholder="Type or paste your message here..."
)

uploaded_file = st.file_uploader("📄 Or upload a text file", type=["txt"])

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- BUTTON ----------------
if st.button("🔍 Analyze Message", use_container_width=True):

    if uploaded_file:
        input_sms = uploaded_file.read().decode()

    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:
        with st.spinner("Analyzing with AI..."):
            transformed = transform_text(input_sms)
            vector = tfidf.transform([transformed])

            prediction = model.predict(vector)[0]
            prob = model.predict_proba(vector)[0]

        st.markdown("---")

        confidence = max(prob)*100

        if prediction == 1:
            st.markdown(
                f"<div class='result-spam'>🚨 SPAM DETECTED<br>Confidence: {confidence:.2f}%</div>",
                unsafe_allow_html=True
            )
            st.error("Avoid links or sharing personal info.")
        else:
            st.markdown(
                f"<div class='result-safe'>✅ SAFE MESSAGE<br>Confidence: {confidence:.2f}%</div>",
                unsafe_allow_html=True
            )
            st.success("This looks safe.")

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Dashboard")

st.sidebar.markdown("### 📌 About")
st.sidebar.info(
    "AI-powered spam detection using NLP and Machine Learning."
)

st.sidebar.markdown("### 🧠 Model Info")
st.sidebar.write("• TF-IDF Vectorizer")
st.sidebar.write("• Naive Bayes Classifier")
st.sidebar.write("• NLP Preprocessing")

st.sidebar.markdown("### 🚀 Developer")
st.sidebar.write("Built by Kalyan Chakraborty")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray;'>© 2026 AI Spam Detector</p>",
    unsafe_allow_html=True
)
