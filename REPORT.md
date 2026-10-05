# One-Page Report - Speech Emotion Detection

**Candidate:** <name>  |  **Links:** GitHub <..> | Dashboard <..>

## What I did
Used RAVDESS speech clips, kept 4 emotions (672 clips; neutral has 96, others 192). Extracted mean and std of 40 MFCCs per clip with librosa.
Split by actor: actors 1-18 for training, 19-24 for testing, so there is no speaker leakage. Trained SVM and Random Forest with balanced class weights,
chose the better one with GroupKFold on training actors, and built a Streamlit dashboard (upload WAV, probability bars, waveform, results page).

## Results (test actors 19-24)
Accuracy: __ | Macro F1: __ | Chosen model: __ | CNN (optional): accuracy __, macro F1 __

## Confusion analysis
<3-4 sentences about your own confusion matrix: which emotions are mixed up and why (e.g. angry vs happy both high-energy; sad vs neutral both low-energy; neutral weaker due to fewer clips).>

## What I would improve with more time
- Add more features (delta MFCC, pitch, energy, chroma) or pretrained audio embeddings (wav2vec2 / HuBERT).
- Data augmentation (noise, pitch shift, time stretch), and use the other RAVDESS emotions / other datasets (CREMA-D, TESS) for more speakers.
- Cross-validate across all 24 actors (leave-actors-out) for a more stable estimate; test on real call-centre audio.
