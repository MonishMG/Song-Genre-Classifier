import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

DATA_DIR = "audio_data"


# we're extracting the features like rhythm, harmony, brightness, tone color, evergy from 
# MFCCs (timbre), Chroma (harmonic structure), Spectral centroid (brightness), zero-crossing rate(sharpness), '
# bandwidth (frequerncy), rolloff (energy), tempo (BPM).
def extract_features(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr, mono=True)

    if y.size == 0:
        raise ValueError("Empty audio signal")

    min_len = 5 * sr
    if len(y) < min_len:
        y = np.pad(y, (0, min_len - len(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)        
    mfcc_mean = mfcc.mean(axis=1)             

    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)      # where the “center of mass” of frequencies is (brightness)
    spec_bw       = librosa.feature.spectral_bandwidth(y=y, sr=sr)     # spread of the spectrum
    zcr           = librosa.feature.zero_crossing_rate(y)              # how often teh signal crosses zero
    rolloff       = librosa.feature.spectral_rolloff(y=y, sr=sr)       # ffrequency below which a certain percentage of total energy is
    spec_centroid_mean = float(spec_centroid.mean())
    spec_bw_mean       = float(spec_bw.mean())
    zcr_mean           = float(zcr.mean())
    rolloff_mean       = float(rolloff.mean())

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)                   # harmonic/pitch classes
    chroma_mean = chroma.mean(axis=1)                        

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)                     # rythm (BPM)
    tempo_val = float(tempo)

    spectral_vec = np.array(                                           # makes a vector of the spectral features + tempo
        [spec_centroid_mean, spec_bw_mean, zcr_mean, rolloff_mean, tempo_val],
        dtype=np.float32
    )

    feature_vector = np.concatenate(                                   #  final feature vector
        [mfcc_mean.astype(np.float32), spectral_vec, chroma_mean.astype(np.float32)],
        axis=0
    )

    if feature_vector.ndim != 1:
        raise ValueError(f"Feature vector not 1-D, got shape {feature_vector.shape}")

    return feature_vector


def load_dataset():
    X = []              # feature vectors
    y = []              # genre labels  

    genres = []
    for genre in os.listdir(DATA_DIR):
        genre_path = os.path.join(DATA_DIR, genre)
        if os.path.isdir(genre_path):
            genres.append(genre)
            for fname in os.listdir(genre_path):
                if not fname.lower().endswith((".mp3", ".wav", ".flac", ".ogg", ".m4a")):
                    continue
                file_path = os.path.join(genre_path, fname)
                try:
                    print(f"Processing  song: {file_path}")
                    features = extract_features(file_path)
                    X.append(features)
                    y.append(genre)
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")

    X = np.array(X)
    y = np.array(y)
    return X, y


def main():
    X, y = load_dataset()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )

    clf = RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)


    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_encoder.classes_
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d")
    plt.title("Confusion Matrix - Audio Genre Classifier")
    plt.tight_layout()
    plt.show()



    joblib.dump(clf, "audio_genre_classifier.pkl")
    joblib.dump(scaler, "audio_scaler.pkl")
    joblib.dump(label_encoder, "audio_label_encoder.pkl")


if __name__ == "__main__":
    main()
