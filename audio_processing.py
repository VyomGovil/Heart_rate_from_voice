import pyloudnorm as pyln
import soundfile as sf
import numpy as np
from scipy.signal import lfilter
import librosa

# --- Pitch-synchronous framing function (shared for both steps) ---
def frame_pitch_synchronous(audio, sr, pitches, hop_length):
    frames = []
    indices = []
    for pitch, idx in zip(pitches, range(0, len(audio), hop_length)):
        if pitch > 0:
            frame_length = int(sr / pitch)
            end_idx = idx + frame_length
            if end_idx <= len(audio):
                frames.append(audio[idx:end_idx])
                indices.append(idx)
    return frames, indices

# --- LPC analysis for glottal and pharynx features ---
def lpc_analysis(frame, order):
    return librosa.lpc(frame.astype(float), order=order)

# --- Glottal/lung-larynx demodulation ---
def iterative_inverse_filtering(audio, sr, lpc_order=16, n_iter=3):
    hop_length = 512
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, hop_length=hop_length)
    pitch_track = pitches[magnitudes.argmax(axis=0), range(magnitudes.shape[1])]
    frames, _ = frame_pitch_synchronous(audio, sr, pitch_track, hop_length)
    glottal_flow = []
    for frame in frames:
        temp_frame = frame
        if len(temp_frame) <= lpc_order + 1:
            continue
        for _ in range(n_iter):
            a = lpc_analysis(temp_frame, lpc_order)
            temp_frame = lfilter(a, 1, temp_frame)
        glottal_flow.append(temp_frame)
    return np.concatenate(glottal_flow) if len(glottal_flow) > 0 else np.array([], dtype=float)

# --- Pharynx demodulation: extract high-order LPC features ---
def pharynx_demodulation(filtered_signal, sr, hop_length=512, lpc_order=30):
    pitches, magnitudes = librosa.piptrack(y=filtered_signal, sr=sr, hop_length=hop_length)
    pitch_track = pitches[magnitudes.argmax(axis=0), range(magnitudes.shape[1])]
    frames, indices = frame_pitch_synchronous(filtered_signal, sr, pitch_track, hop_length)
    pharynx_features = []
    for frame in frames:
        if len(frame) > lpc_order + 1:
            lpc_coeffs = librosa.lpc(frame.astype(float), order=lpc_order)
            pharynx_features.append(lpc_coeffs)
        else:
            pharynx_features.append(np.zeros(lpc_order+1))
    pharynx_features = np.stack(pharynx_features)
    return pharynx_features, indices

# --- Load and normalize audio ---
data, rate = sf.read('voice_sample.wav')
meter = pyln.Meter(rate)
loudness = meter.integrated_loudness(data)
target_loudness = -12.0
print(f"Original loudness: {loudness:.2f} LUFS")
normalized_audio = pyln.normalize.loudness(data, loudness, target_loudness)
"""
# --- Lip vibration removal ---
alpha = 0.97
b = np.array([1, -alpha])
a = np.array([1])
filtered_signal = lfilter(b, a, normalized_audio)

# --- Lung-Larynx Filtering (Glottal Flow Extraction) ---
glottal_waveform = iterative_inverse_filtering(filtered_signal, rate)

# --- Pharynx Demodulation Step ---
pharynx_features, frame_indices = pharynx_demodulation(glottal_waveform, rate)

# pharynx_features: each row is a vector of LPC coefficients for one pitch-synchronous frame,
# suitable for use in downstream cardiac demodulation modeling.

print(pharynx_features.shape)
print(pharynx_features)
"""