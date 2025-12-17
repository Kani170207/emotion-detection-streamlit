import streamlit as st
import pickle
import re
# ===============================
# 🎨 Custom CSS for Fancy UI
# ===============================
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Text area */
textarea {
    border-radius: 15px !important;
    font-size: 16px !important;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #ff512f, #dd2476);
    color: white;
    font-size: 18px;
    padding: 10px 25px;
    border-radius: 30px;
    border: none;
    transition: 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #dd2476, #ff512f);
}

/* Success message */
.stAlert {
    border-radius: 15px;
}

/* Footer */
.footer {
    text-align: center;
    opacity: 0.7;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)


# ===============================
# 🎭 Emotion → Emoji mapping
# ===============================
emotion_emoji = {
    "happy": "😄",
    "love": "❤️",
    "sad": "😢",
    "angry": "😡",
    "fear": "😨",
    "neutral": "😐"
}

# ===============================
# 🎨 Emotion → Color mapping
# ===============================
emotion_color = {
    "happy": "#2ecc71",
    "love": "#e84393",
    "sad": "#3498db",
    "angry": "#e74c3c",
    "fear": "#f39c12",
    "neutral": "#95a5a6"
}

# ===============================
# 📦 Load Model & Vectorizer
# ===============================
model = pickle.load(open("emotion_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ===============================
# 🧹 Text Cleaning (SAFE VERSION)
# ===============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

# ===============================
# 🖥 Streamlit UI
# ===============================
st.set_page_config(page_title="Emotion Detection App", layout="centered")

# ===============================
# 🎭 Fancy Title & Subtitle
# ===============================
st.markdown(
    "<h1 style='text-align:center;'>🎭 Emotion Detection from Text</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; font-size:18px;'>"
    "Detect human emotions using <b>NLP & Machine Learning</b>"
    "</p>",
    unsafe_allow_html=True
)

# ===============================
# ✍️ Text Area Input (STEP 7)
# ===============================
user_text = st.text_area(
    "✍️ Enter your text here:",
    placeholder="Example: I am very happy today 😊",
    height=150
)

# ===============================
# 🔮 Prediction Button
# ===============================
if st.button("Predict Emotion"):
    if user_text.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean_text(user_text)
        vector = vectorizer.transform([cleaned])

        # Predict emotion
        prediction = model.predict(vector)[0]

        # Get emoji & color
        emoji = emotion_emoji.get(prediction.lower(), "🙂")
        color = emotion_color.get(prediction.lower(), "#34495e")

        # Text result
        st.success(f"Detected Emotion: **{prediction.upper()}** {emoji}")

        # Big emoji
        st.markdown(
            f"<h1 style='text-align: center;'>{emoji}</h1>",
            unsafe_allow_html=True
        )

        # Colored emotion card
        st.markdown(
            f"""
            <div style="
                background-color:{color};
                padding:20px;
                border-radius:15px;
                text-align:center;
                color:white;
                font-size:22px;
                font-weight:bold;">
                {emoji} {prediction.upper()}
            </div>
            """,
            unsafe_allow_html=True
        )

# ===============================
# 📌 Footer
# ===============================
st.markdown(
    "<hr><center>🚀 Built with Streamlit & Machine Learning</center>",
    unsafe_allow_html=True
)
