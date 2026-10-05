# Speech Emotion Detection (RAVDESS)

Predicts **angry / happy / sad / neutral** from short speech clips. Built for the Hexovate AI/ML internship assessment (Task 3).

**Live dashboard:** https://speech-emotion-detection-bladsbyyqyhhcpe7uspxtk.streamlit.app/

## Data
RAVDESS speech audio, 4 emotions only: 672 clips (192 angry, 192 happy, 192 sad, 96 neutral).
Emotion = 3rd number in the file name (01 neutral, 03 happy, 04 sad, 05 angry); actor = last number.

## Method
1. Explore: class counts, waveform and mel spectrogram per emotion.
2. Features: 40 MFCCs per clip (librosa, 22,050 Hz), mean + std = 80 features.
3. **Actor-based split:** actors 1-18 train, actors 19-24 test (voices never seen in training).
4. Models: SVM (RBF) and Random Forest, both with `class_weight="balanced"` because neutral has half the clips. Model selection used GroupKFold on training actors only.
5. Metrics: accuracy, macro F1, confusion matrix (see `metrics.json`).

## Results
| Model | Accuracy | Macro F1 |
|---|---|---|
| Random Forest (chosen, balanced class weights) | 66.7% | 0.614 |

Train: actors 1-18 (504 clips). Test: actors 19-24 (168 clips, unseen voices). SVM cross-validation macro F1 was 0.476 and Random Forest was 0.494, so Random Forest was chosen.

Per-class F1 on test actors: angry 0.81, sad 0.67, happy 0.65, neutral 0.33.

## Confusion analysis
The model recognises **angry** best (F1 0.81, 39 of 48 clips correct), because angry speech is loud and has a distinctive energy pattern. **Neutral** is the weakest class (recall 0.25, F1 0.33): 14 of the 24 neutral test clips were predicted as sad, since both are quiet, low-energy and flat in tone. Neutral also has only half as many training clips as the other emotions, even with balanced class weights. **Happy** was confused with sad (9 clips) and with angry (7 clips), because happy and angry share high pitch and energy. The test set is small (168 clips, only 24 neutral), so these numbers could change with other actors.

## Files
- `emotion_detection.ipynb` - full pipeline
- `app.py` - Streamlit dashboard (Predict page + Results page)
- `model.joblib`, `metrics.json`, `confusion_matrix.png`, `features.csv` - outputs of the notebook
- `requirements.txt`

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
