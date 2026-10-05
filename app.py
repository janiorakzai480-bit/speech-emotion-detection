import io, json, os
import numpy as np, pandas as pd
import joblib, librosa, librosa.display
import matplotlib.pyplot as plt
import streamlit as st

SR = 22050
LABELS = ["angry", "happy", "sad", "neutral"]
st.set_page_config(page_title="Speech Emotion Detection", page_icon="🎙️", layout="wide")

# --- must be identical to the notebook ---
def load_audio_bytes(data):
    y, _ = librosa.load(io.BytesIO(data), sr=SR, duration=4.0)
    y, _ = librosa.effects.trim(y, top_db=25)
    return y

def extract_features(y, sr=SR):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

page = st.sidebar.radio("Page", ["Predict emotion", "Model results"])

if page == "Predict emotion":
    st.title("🎙️ Speech Emotion Detection")
    st.write("Upload a short WAV speech clip. The model predicts one of 4 emotions: angry, happy, sad, neutral.")
    f = st.file_uploader("Upload WAV file", type=["wav"])
    if f is not None:
        data = f.read()
        st.audio(data, format="audio/wav")
        try:
            y = load_audio_bytes(data)
            if len(y) < SR * 0.3:
                st.warning("Audio is too short or silent. Try a clip with speech.")
            else:
                bundle = load_model()
                model, classes = bundle["model"], list(bundle["labels"])
                probs = model.predict_proba(extract_features(y).reshape(1, -1))[0]
                s = pd.Series(probs, index=classes).reindex(LABELS)
                st.subheader(f"Predicted emotion: **{s.idxmax().upper()}** ({s.max():.0%})")
                c1, c2 = st.columns(2)
                with c1:
                    st.write("Probability per emotion")
                    st.bar_chart(s)
                with c2:
                    st.write("Waveform")
                    fig, ax = plt.subplots(figsize=(6, 3))
                    librosa.display.waveshow(y, sr=SR, ax=ax)
                    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
                    st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not read this file: {e}")
else:
    st.title("📊 Model results")
    if not os.path.exists("metrics.json"):
        st.error("metrics.json not found. Run the notebook first.")
    else:
        m = json.load(open("metrics.json"))
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{m['accuracy']:.1%}")
        c2.metric("Macro F1", f"{m['macro_f1']:.3f}")
        c3.metric("Model", m["model"])
        st.caption(f"Train actors: {m['train_actors']} ({m['n_train']} clips) | Test actors: {m['test_actors']} ({m['n_test']} clips, unseen voices)")
        a, b = st.columns(2)
        with a:
            st.subheader("Class counts")
            st.bar_chart(pd.Series(m["class_counts"]).reindex(LABELS))
        with b:
            st.subheader("Confusion matrix")
            cm = pd.DataFrame(m["confusion_matrix"], index=[f"true {l}" for l in m["labels"]],
                              columns=[f"pred {l}" for l in m["labels"]])
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(cm.values, cmap="Blues")
            ax.set_xticks(range(4)); ax.set_xticklabels(m["labels"]); ax.set_yticks(range(4)); ax.set_yticklabels(m["labels"])
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, cm.values[i, j], ha="center", va="center")
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            st.pyplot(fig)
        st.subheader("Per-class report")
        rep = pd.DataFrame(m["report"]).T.loc[m["labels"] + ["macro avg"], ["precision", "recall", "f1-score", "support"]]
        st.dataframe(rep.round(3))
