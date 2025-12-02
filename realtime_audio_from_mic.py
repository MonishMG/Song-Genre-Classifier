import numpy as np
import sounddevice as sd
import librosa
import joblib

DURATION = 5      # seconds to listen each time
SR = 22050        # sampling rate for librosa / recording

# Load model artifacts
clf = joblib.load("audio_genre_classifier.pkl")
scaler = joblib.load("audio_scaler.pkl")
label_encoder = joblib.load("audio_label_encoder.pkl")

# we're extracting the features like rhythm, harmony, brightness, tone color, evergy from 
# MFCCs (timbre), Chroma (harmonic structure), Spectral centroid (brightness), zero-crossing rate(sharpness), '
# bandwidth (frequerncy), rolloff (energy), tempo (BPM).
def extract_features_from_signal(y, sr=SR):
    # Guard against empty audio
    if y.size == 0:
        raise ValueError("Empty audio signal")

    # Ensure at least 3 seconds (pad if shorter)
    min_len = 3 * sr
    if len(y) < min_len:
        y = np.pad(y, (0, min_len - len(y)))

    # --- MFCCs (20) ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)        # (20, T)
    mfcc_mean = mfcc.mean(axis=1)                             # (20,)

    # --- Spectral features (scalars) ---
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)      # (1, T)
    spec_bw       = librosa.feature.spectral_bandwidth(y=y, sr=sr)     # (1, T)
    zcr           = librosa.feature.zero_crossing_rate(y)              # (1, T)
    rolloff       = librosa.feature.spectral_rolloff(y=y, sr=sr)       # (1, T)

    spec_centroid_mean = float(spec_centroid.mean())
    spec_bw_mean       = float(spec_bw.mean())
    zcr_mean           = float(zcr.mean())
    rolloff_mean       = float(rolloff.mean())

    # --- Chroma (12) ---
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)          # (12, T)
    chroma_mean = chroma.mean(axis=1)                         # (12,)

    # --- Tempo (BPM) ---
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_val = float(tempo)

    spectral_vec = np.array(
        [spec_centroid_mean, spec_bw_mean, zcr_mean, rolloff_mean, tempo_val],
        dtype=np.float32
    )

    feature_vector = np.concatenate(
        [mfcc_mean.astype(np.float32), spectral_vec, chroma_mean.astype(np.float32)],
        axis=0
    )

    if feature_vector.ndim != 1:
        raise ValueError(f"Feature vector not 1-D, got shape {feature_vector.shape}")

    return feature_vector

def record_and_classify():
    print(f"\nRecording {DURATION} seconds of audio... (make sure music is playing)")
    audio = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    y = audio.flatten()

    features = extract_features_from_signal(y)
    X_scaled = scaler.transform([features])

    pred = clf.predict(X_scaled)[0]
    genre = label_encoder.inverse_transform([pred])[0]

    print(f"🎵 Predicted genre: {genre}")



def main():
    print("Real-time audio genre classifier")
    print("Press Ctrl+C to stop.\n")
    try:
        while True:
            record_and_classify()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
