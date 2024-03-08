import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
# Load the audio file
audio_file = 'audio.wav'
# audio, sr = librosa.load(audio_file)
# sr = 44100
# once you fix the sample rate we will have fixed output shape.
# sr = 2800
time_sec = 2
channels = 2
n_mels = 256
time_length = 256
hop_length = 256
sr = hop_length * (time_length - 1) // time_sec
# sr = hop_length * time_length // time_sec
# hop_length = 512
# n_mels = 128
# audio = np.random.random((sr*time_sec, channels))
audio = np.random.random((channels, sr*time_sec))

# Convert the audio to a mel spectrogram
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels = n_mels, hop_length = hop_length)
# (2, 128, 87)
# Convert to log scale (dB)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
# (2, 128, 87)
# channel, deg
# print(mel_spec_db)
# print(mel_spec)
print(mel_spec_db.shape)
print(mel_spec.shape) # (2, 256, 256)
print(sr)
# getting blind. getting used to it.
# at least we want some statistics.
# we have fixed the shape. now what?