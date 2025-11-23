import sounddevice as sd
import serial
import numpy as np
import pyloudnorm as pyln
import librosa
from scipy.signal import lfilter
import time
import threading
import os

# --- Import the processing functions from your teammate's file ---
try:
    from audio_processing import iterative_inverse_filtering, pharynx_demodulation
except ImportError:
    print("Error: Could not find 'audio_processing.py'.")
    print("Please make sure 'audio_processing.py' is in the same folder as this script.")
    exit()

# --- Configuration ---
SERIAL_PORT = 'COM3'  # <-- IMPORTANT: CHANGE THIS to your Arduino's port
BAUD_RATE = 9600
SAMPLE_RATE = 48000          # (Hz) From paper
DURATION = 60                # (seconds) Duration of recording
LPC_ORDER = 30               # (Pharynx model order) From paper
TARGET_LOUDNESS = -12.0      # (LUFS) From paper
ENHANCEMENT_COEFF = 0.97     # (alpha) From paper

# --- Global lists to store data from threads ---
audio_frames = []
ecg_data_list = []

# --- 1. Audio and ECG Recording Functions (Run in Threads) ---

def record_audio(stop_event):
    """
    Records audio from the default microphone until stop_event is set.
    """
    global audio_frames
    audio_frames = []
    print("[Audio] Recording started...")
    
    def callback(indata, frames, time, status):
        if status:
            print(status, flush=True)
        audio_frames.append(indata.copy())

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=callback):
            while not stop_event.is_set():
                sd.sleep(100) # Wait in 100ms intervals
    except sd.PortAudioError as e:
        print(f"\n[Audio] Error: Could not open audio device. {e}")
        print("Please check if your microphone is connected.")
    
    print("[Audio] Recording stopped.")

def record_ecg(port, baud, stop_event):
    """
    Reads data from the serial port until stop_event is set.
    """
    global ecg_data_list
    ecg_data_list = []
    
    try:
        ser = serial.Serial(port, baud, timeout=1)
        print(f"[ECG] Connecting to {port}...")
        time.sleep(2) # Wait for connection to establish
        ser.flushInput()
        print("[ECG] Connection successful. Reading data...")
        
        while not stop_event.is_set():
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8').strip()
                    if line == '!':
                        print("[ECG] Warning: Leads off detected!")
                    elif line:
                        # --- FIX 1: Changed int(line) to float(line) ---
                        ecg_data_list.append(float(line))
                        # ---------------------------------------------
                except Exception as e:
                    # Catch malformed lines, etc.
                    print(f"[ECG] Error reading line: {e}", flush=True)
                        
    except serial.SerialException as e:
        print(f"\n[ECG] Error: Could not open port {port}. {e}")
        print("Please check your SERIAL_PORT variable and Arduino connection.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
        print("[ECG] Serial connection closed.")

# --- 2. Audio Processing Functions are in audio_processing.py ---
# (They are imported at the top of this file)

# --- 3. Main Execution ---

def main():
    global audio_frames, ecg_data_list

    output_folder = "test_dataset"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created new folder: {output_folder}")
    
    output_filename = input(f"Enter a name for this data sample (e.g., rest_01): ")
    if not output_filename:
        print("Invalid name. Exiting.")
        return
    output_filepath = os.path.join(output_folder, f"{output_filename}.npz")
    #output_filepath = f"{output_filename}.npz"
    
    stop_event = threading.Event()
    
    # Start the recording threads
    audio_thread = threading.Thread(target=record_audio, args=(stop_event,))
    ecg_thread = threading.Thread(target=record_ecg, args=(SERIAL_PORT, BAUD_RATE, stop_event))
    
    audio_thread.start()
    ecg_thread.start()
    
    print(f"\nRecording for {DURATION} seconds... Speak into your microphone.")
    try:
        time.sleep(DURATION)
    except KeyboardInterrupt:
        print("\nStopping recording early.")
    
    # Signal threads to stop
    stop_event.set()
    
    # Wait for threads to finish
    audio_thread.join()
    ecg_thread.join()
    
    print("\n--- Recording Complete ---")
    
    # --- Process Data ---
    
    # 1. Process Audio
    if not audio_frames:
        print("Error: No audio was recorded. Exiting.")
        return
        
    audio_data = np.concatenate(audio_frames, axis=0).flatten()
    print(f"Audio data shape: {audio_data.shape}")

    print("Processing audio...")
    # Step 1. Normalize Loudness
    meter = pyln.Meter(SAMPLE_RATE)
    loudness = meter.integrated_loudness(audio_data)

    # --- FIX 2: Check for silent audio ---
    if not np.isfinite(loudness):
        print(f"Error: Could not calculate loudness (audio silent? Loudness: {loudness}). Exiting.")
        return
    # -------------------------------------

    normalized_audio = pyln.normalize.loudness(audio_data, loudness, TARGET_LOUDNESS)
    
    # Step 2. Lip Vibration Removal (Enhancement)
    filtered_signal = lfilter([1, -ENHANCEMENT_COEFF], [1], normalized_audio)
    
    # Step 3. Glottal Flow Extraction (Lung-Larynx)
    # This function is now imported from audio_processing.py
    glottal_waveform = iterative_inverse_filtering(filtered_signal, SAMPLE_RATE)
    
    if glottal_waveform.shape[0] == 0:
        print("Error: Could not extract glottal waveform. (No pitch detected?) Exiting.")
        return

    # Step 4. Pharynx Demodulation (Get final X features)
    # This function is also imported from audio_processing.py
    pharynx_features, _ = pharynx_demodulation(glottal_waveform, SAMPLE_RATE, lpc_order=LPC_ORDER)
    
    if pharynx_features.shape[0] == 0:
        print("Error: Could not extract pharynx features. Exiting.")
        return
        
    print(f"Pharynx features shape (X): {pharynx_features.shape}")

    # 2. Process ECG
    if not ecg_data_list:
        print("Error: No ECG data was recorded. Exiting.")
        return
        
    ecg_data_np = np.array(ecg_data_list, dtype=np.float32)
    print(f"ECG data shape (Y): {ecg_data_np.shape}")

    # 3. Save to .npz file
    np.savez_compressed(
        output_filepath,
        pharynx_features=pharynx_features,
        ecg_data=ecg_data_np
    )
    
    print(f"\n--- Success! ---")
    print(f"Data saved to {output_filepath}")
    print(f"This file is now ready for the ML training script.")

if __name__ == "__main__":
    main()