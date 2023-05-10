import pyaudio
import wave

chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 2
fs = 44100  # Record at 44100 samples per second
# seconds = 3
# no seconds limit.
# filename = "output.wav"
from utils import filepaths, check_redis_on, check_redis_off, TimestampedContext

p = pyaudio.PyAudio()  # Create an interface to PortAudio

# sudo modprobe snd-aloop

# use loopback device 0?

input_device_index = 2

print("Recording")

stream = p.open(
    format=sample_format,
    channels=channels,  # this is microphone. how to record internal audio?
    rate=fs,
    frames_per_buffer=chunk,  # this is the chunk.
    input_device_index=input_device_index,  # macos: 0 for blackhole. but you must set blackhole as output.
    input=True,
)

# frames = []  # Initialize array to store frames

wf = wave.open(filepaths.audio_record, "wb")
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(fs)

# Store data in chunks for 3 seconds
# for i in range(0, int(fs / chunk * seconds)):
if check_redis_on():
    with TimestampedContext(filepaths.audio_timestamps) as t:
        while check_redis_off() is False:
            data = stream.read(chunk)
            wf.writeframes(data)
            # wf.writeframes(b"".join(frames))
            # frames.append(data)

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        # Terminate the PortAudio interface
        p.terminate()

        print("Finished recording")

        # Save the recorded data as a WAV file
        wf.close()
        print("Saved audio recording to: {}".format(filepaths.audio_record))
else:
    print("AudioRecorder: Can't start. Redis signal is off/".upper())
