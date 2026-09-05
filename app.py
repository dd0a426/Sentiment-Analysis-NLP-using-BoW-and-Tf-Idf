import streamlit as st
import joblib
import re
import string
import pandas as pd
from pathlib import Path
from datetime import datetime


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Emotion Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# EMOTION LABELS, EMOJIS & THEME COLORS
# ============================================================

EMOTION_LABELS = {
    0: "sadness",
    1: "anger",
    2: "love",
    3: "surprise",
    4: "fear",
    5: "joy"
}

EMOTION_EMOJI = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😠",
    "love": "❤️",
    "surprise": "😮",
    "fear": "😨"
}

# A distinct accent color per emotion, used for result cards & charts
EMOTION_COLOR = {
    "joy": "#F5B60A",
    "sadness": "#4A7FE0",
    "anger": "#E5484D",
    "love": "#E0559C",
    "surprise": "#9B6DF0",
    "fear": "#6B7280"
}


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    .hero {
        text-align: center;
        padding: 10px 0 25px 0;
    }

    .hero-title {
        font-size: 46px;
        font-weight: 800;
        background: linear-gradient(90deg, #6D5EF0, #E0559C, #F5B60A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }

    .hero-subtitle {
        color: #7A7A85;
        font-size: 17px;
    }

    .card {
        background-color: #FFFFFF08;
        border: 1px solid rgba(150,150,150,0.18);
        border-radius: 18px;
        padding: 22px 26px;
        margin-bottom: 18px;
    }

    .result-card {
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin-top: 10px;
        border: 1px solid rgba(150,150,150,0.18);
    }

    .result-label {
        font-size: 15px;
        letter-spacing: 1px;
        text-transform: uppercase;
        opacity: 0.75;
    }

    .result-emotion {
        font-size: 44px;
        font-weight: 800;
        margin-top: 6px;
    }

    .stat-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        background: rgba(150,150,150,0.15);
        font-size: 13px;
        margin-right: 8px;
    }

    .prob-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 6px 0;
    }

    .prob-label {
        width: 100px;
        font-size: 14px;
        text-align: right;
    }

    .prob-bar-bg {
        flex: 1;
        height: 14px;
        border-radius: 999px;
        background: rgba(150,150,150,0.15);
        overflow: hidden;
    }

    .prob-bar-fill {
        height: 100%;
        border-radius: 999px;
    }

    .prob-value {
        width: 55px;
        font-size: 13px;
        opacity: 0.8;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "logistic_model_tfidf.joblib"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.joblib"


# ============================================================
# LOAD STOPWORDS
# ============================================================

@st.cache_data
def load_stopwords():
    try:
        import nltk
        from nltk.corpus import stopwords

        try:
            words = stopwords.words("english")
        except LookupError:
            nltk.download("stopwords", quiet=True)
            words = stopwords.words("english")

        return set(words)

    except Exception:
        return set()


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):
    stop_words = load_stopwords()

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = "".join(char for char in text if char.isascii())

    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]

    return " ".join(cleaned_words)


# ============================================================
# LOAD MODEL AND VECTORIZER
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found:\n{MODEL_PATH}")
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"TF-IDF vectorizer not found:\n{VECTORIZER_PATH}")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_emotion(text, model, vectorizer):
    cleaned_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_tfidf)
    prediction_numeric = int(prediction[0])
    predicted_emotion = EMOTION_LABELS.get(prediction_numeric, "Unknown")

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(text_tfidf)[0]

    return predicted_emotion, probabilities, cleaned_text


def build_probability_map(model, probabilities):
    probability_map = {}
    for cls, probability in zip(model.classes_, probabilities):
        label = EMOTION_LABELS.get(int(cls), str(cls))
        probability_map[label] = float(probability)
    return dict(sorted(probability_map.items(), key=lambda kv: kv[1], reverse=True))


def render_probability_bars(probability_map):
    for emotion, prob in probability_map.items():
        color = EMOTION_COLOR.get(emotion, "#888")
        emoji = EMOTION_EMOJI.get(emotion, "")
        st.markdown(
            f"""
            <div class="prob-row">
                <div class="prob-label">{emoji} {emotion.title()}</div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{prob*100:.1f}%; background:{color};"></div>
                </div>
                <div class="prob-value">{prob:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🧠 Emotion Detection</div>
        <div class="hero-subtitle">
            NLP-based emotion classification using TF-IDF + Logistic Regression
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("📌 About")
    st.write(
        "This app predicts the emotion expressed in a piece of text "
        "using a TF-IDF + Logistic Regression model."
    )

    st.divider()
    st.subheader("Supported Emotions")

    cols = st.columns(2)
    emotion_items = list(EMOTION_EMOJI.items())
    for i, (emo, emoji) in enumerate(emotion_items):
        with cols[i % 2]:
            st.markdown(
                f"<span class='stat-pill'>{emoji} {emo.title()}</span>",
                unsafe_allow_html=True
            )

    st.divider()
    st.subheader("Model Details")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Features:** TF-IDF")

    st.divider()
    if st.button("🗑️ Clear history", use_container_width=True):
        st.session_state["history"] = []
        st.rerun()


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model, vectorizer = load_model()
    st.toast("Model loaded successfully", icon="✅")
except Exception as error:
    st.error("❌ Model loading failed")
    st.code(str(error))
    st.info(
        "Make sure these files are in the same folder as app.py:\n\n"
        "- logistic_model_tfidf.joblib\n"
        "- tfidf_vectorizer.joblib"
    )
    st.stop()


if "history" not in st.session_state:
    st.session_state["history"] = []


# ============================================================
# TABS
# ============================================================

tab_single, tab_batch, tab_history = st.tabs(
    ["✍️ Single Text", "📄 Batch Analysis", "🕘 History"]
)


# ------------------------------------------------------------
# TAB 1: SINGLE TEXT
# ------------------------------------------------------------

with tab_single:

    examples = {
        "Select an example": "",
        "Joy": "I am extremely happy and excited about my new job.",
        "Sadness": "I feel very lonely and disappointed today.",
        "Anger": "I am extremely angry about what happened.",
        "Love": "I love my family and I feel very grateful for them.",
        "Fear": "I am scared and worried about what might happen.",
        "Surprise": "I was completely shocked by the unexpected news."
    }

    col_ex, col_clear = st.columns([4, 1])
    with col_ex:
        selected_example = st.selectbox("💡 Try an example", list(examples.keys()))
    with col_clear:
        st.write("")
        st.write("")

    default_text = examples[selected_example] if selected_example != "Select an example" else ""

    user_text = st.text_area(
        "Enter your text",
        value=default_text,
        height=150,
        placeholder="Example: I am very happy today!"
    )

    char_count = len(user_text)
    word_count = len(user_text.split()) if user_text.strip() else 0

    st.markdown(
        f"<span class='stat-pill'>🔤 {word_count} words</span>"
        f"<span class='stat-pill'>🔠 {char_count} characters</span>",
        unsafe_allow_html=True
    )

    predict_button = st.button(
        "🔍 Predict Emotion", type="primary", use_container_width=True
    )

    if predict_button:
        if not user_text.strip():
            st.warning("⚠️ Please enter some text first.")
        else:
            with st.spinner("Analyzing your text..."):
                emotion, probabilities, cleaned_text = predict_emotion(
                    user_text, model, vectorizer
                )

            emoji = EMOTION_EMOJI.get(emotion, "🧠")
            color = EMOTION_COLOR.get(emotion, "#888")

            st.markdown(
                f"""
                <div class="result-card" style="background: {color}18; border-color: {color}55;">
                    <div class="result-label">Predicted Emotion</div>
                    <div class="result-emotion" style="color:{color};">{emoji} {emotion.title()}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            probability_map = None
            confidence = None

            if probabilities is not None:
                probability_map = build_probability_map(model, probabilities)
                confidence = probability_map.get(emotion, 0)

                st.write("")
                st.metric("🎯 Prediction Confidence", f"{confidence:.2%}")

                st.subheader("📊 Emotion Probabilities")
                render_probability_bars(probability_map)

            with st.expander("🔎 View Processed Text"):
                st.write(cleaned_text if cleaned_text else "No text remained after preprocessing.")

            st.session_state["history"].append(
                {
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Text": user_text,
                    "Emotion": emotion,
                    "Confidence": f"{confidence:.2%}" if confidence is not None else "N/A"
                }
            )


# ------------------------------------------------------------
# TAB 2: BATCH ANALYSIS
# ------------------------------------------------------------

with tab_batch:

    st.write(
        "Upload a CSV file with a column of text, or paste multiple lines below "
        "(one text per line), to classify emotions in bulk."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    pasted_text = st.text_area(
        "...or paste one text per line",
        height=150,
        placeholder="I am so happy today!\nThis makes me really angry.\n..."
    )

    text_column = None
    batch_texts = []

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            text_column = st.selectbox("Select the text column", df_upload.columns)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    run_batch = st.button("🚀 Run Batch Prediction", use_container_width=True)

    if run_batch:
        if uploaded_file is not None and text_column is not None:
            batch_texts = df_upload[text_column].astype(str).tolist()
        elif pasted_text.strip():
            batch_texts = [line for line in pasted_text.split("\n") if line.strip()]

        if not batch_texts:
            st.warning("⚠️ Please upload a CSV or paste some text first.")
        else:
            with st.spinner(f"Analyzing {len(batch_texts)} texts..."):
                results = []
                for txt in batch_texts:
                    emo, probs, _ = predict_emotion(txt, model, vectorizer)
                    conf = None
                    if probs is not None:
                        pm = build_probability_map(model, probs)
                        conf = pm.get(emo, 0)
                    results.append(
                        {
                            "Text": txt,
                            "Predicted Emotion": f"{EMOTION_EMOJI.get(emo,'')} {emo.title()}",
                            "Confidence": f"{conf:.2%}" if conf is not None else "N/A"
                        }
                    )

            results_df = pd.DataFrame(results)
            st.success(f"✅ Analyzed {len(results_df)} texts")
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download results as CSV",
                data=csv_bytes,
                file_name="emotion_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )


# ------------------------------------------------------------
# TAB 3: HISTORY
# ------------------------------------------------------------

with tab_history:

    if not st.session_state["history"]:
        st.info("No predictions yet. Try analyzing some text in the Single Text tab!")
    else:
        history_df = pd.DataFrame(st.session_state["history"]).iloc[::-1]
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        csv_bytes = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download history as CSV",
            data=csv_bytes,
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("NLP Emotion Classification | TF-IDF + Logistic Regression | Streamlit")        


