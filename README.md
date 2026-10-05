# Speech Emotion Detection (RAVDESS)

Predicts **angry / happy / sad / neutral** from short speech clips. Built for the Hexovate AI/ML internship assessment (Task 3).

**Live dashboard:** <PASTE LINK HERE>

## Data
RAVDESS speech audio, 4 emotions only -> 672 clips (192 angry, 192 happy, 192 sad, 96 neutral).
Emotion = 3rd number in the file name (01 neutral, 03 happy, 04 sad, 05 angry); actor = last number.

## Method
1. Explore: class counts, waveform and mel spectrogram per emotion.
2. Features: 40 MFCCs per clip (librosa, 22,050 Hz), mean + std -> 80 features.
3. **Actor-based split:** actors 1-18 train, actors 19-24 test (voices never seen in training).
4. Models: SVM (RBF) and Random Forest, both with `class_weight="balanced"` because neutral has half the clips.
   Model selection used GroupKFold on training actors only.
5. Metrics: accuracy, macro F1, confusion matrix (see `metrics.json`).
6. Optional: small CNN on mel spectrograms (last notebook section).

## Results
| Model | Accuracy | Macro F1 |
|---|---|---|
| (fill from notebook) | | |

## Files
- `emotion_detection.ipynb` - full pipeline
- `app.py` - Streamlit dashboard (Predict page + Results page)
- `model.joblib`, `metrics.json`, `confusion_matrix.png` - outputs of the notebook
- `requirements.txt`

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
Deploy: push all files to GitHub -> https://share.streamlit.io -> New app -> select repo and `app.py`.
