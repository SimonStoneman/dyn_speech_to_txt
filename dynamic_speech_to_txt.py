import tkinter as tk
import threading
import sounddevice as sd
import numpy as np
import whisper
import scipy.io.wavfile as wav
import time

# Parameters
sample_rate = 16000
chunk_duration = 5  # seconds
chunk_samples = int(chunk_duration * sample_rate)

stop_event = threading.Event()  # Event to signal when to stop

# Load the Whisper model once
model = whisper.load_model("base")

def record_and_transcribe():
    """Continuously record audio chunks and transcribe until stop_event is set."""
    try:
        while not stop_event.is_set():
            # Record a chunk of audio
            print(f"Recording for {chunk_duration} seconds...")
            audio = sd.rec(chunk_samples, samplerate=sample_rate, channels=1, dtype='float32')
            sd.wait()

            # Save to file
            wav.write("temp_chunk.wav", sample_rate, audio)

            # Transcribe the chunk
            print("Transcribing chunk...")
            result = model.transcribe("temp_chunk.wav")
            text = result["text"]
            print("Transcription:", text)

            # Update GUI text area (append new text)
            output_text.configure(state='normal')
            output_text.insert(tk.END, text + "\n")
            output_text.configure(state='disabled')

            # Optional: small sleep if needed
            # time.sleep(0.1)

    except Exception as e:
        print("Error in recording thread:", e)

def start_recording():
    """Start the recording thread."""
    # Make sure the stop event is cleared before starting
    stop_event.clear()
    # Start the background thread
    thread = threading.Thread(target=record_and_transcribe, daemon=True)
    thread.start()

def stop_recording():
    """Signal the recording thread to stop."""
    stop_event.set()

# Create the GUI
root = tk.Tk()
root.title("Whisper Continuous Recording")

start_button = tk.Button(root, text="Start Recording", command=start_recording)
start_button.pack(pady=10)

stop_button = tk.Button(root, text="Stop Recording", command=stop_recording)
stop_button.pack(pady=10)

output_text = tk.Text(root, state='disabled', width=50, height=10)
output_text.pack(pady=10)

root.mainloop()