# One-Page Report - Speech Emotion Detection

**Candidate:** Sajid Rehman  |  **Links:** GitHub https://github.com/janiorakzai480-bit/speech-emotion-detection| Dashboard https://speech-emotion-detection-bladsbyyqyhhcpe7uspxtk.streamlit.app/

## What I did
I used RAVDESS speech clips and kept 4 emotions (672 clips; neutral has 96, the others 192 each). I extracted the mean and standard deviation of 40 MFCCs per clip with librosa. I split by actor: actors 1-18 for training (504 clips) and actors 19-24 for testing (168 clips), so the model is tested on unseen voices. I trained an SVM and a Random Forest with balanced class weights and chose the better one with GroupKFold on the training actors (SVM 0.476, Random Forest 0.494 macro F1). I then built a Streamlit dashboard with an upload and prediction page (probability bars and waveform) and a results page.

## Results (test actors 19-24)
Accuracy 66.7%, macro F1 0.614 (Random Forest). Per-class F1: angry 0.81, sad 0.67, happy 0.65, neutral 0.33

## Confusion analysis
The model recognises angry best (39 of 48 correct) because angry speech is loud and distinctive. Neutral is the weakest class (recall 0.25): 14 of 24 neutral clips were predicted as sad, because both are quiet and low-energy, and neutral has half as many training clips. Happy is confused with sad (9 clips) and angry (7 clips). The test set is small, so results could change with other actors.

## What I would improve with more time
-  Neutral is the main weakness, so I would add more neutral data or augmentation (noise, pitch shift, time stretch).
- Add more features (delta MFCC, pitch, energy) or pretrained audio embeddings such as wav2vec2 or HuBERT.
- ross-validate across all 24 actors for a more stable estimate, and test on real call-centre audio.
- Train the CNN on mel spectrograms and compare it with this model.
