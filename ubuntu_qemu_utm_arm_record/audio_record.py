import parse
from pyaudio_get_device_info import get_audio_input_device_list
import platform
from utils import (
    filepaths,
    check_redis_on,
    check_redis_off,
    TimestampedContext,
    set_redis_off_on_exception,
)
import pyaudio
import wave

set_redis_off_on_exception()

chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 2
fs = 44100  # Record at 44100 samples per second
# seconds = 3
# no seconds limit.
# filename = "output.wav"

p = pyaudio.PyAudio()  # Create an interface to PortAudio

# sudo modprobe snd-aloop

# use loopback device 0?
system = platform.system()


input_device_list = get_audio_input_device_list(p)


def get_input_device_index(input_device_list, pattern):
    for index, device_name in input_device_list:
        if parse.parse(pattern, device_name):
            print("SELECT AUDIO INPUT DEVICE: %s" % device_name)
            return index
    raise Exception("Cannot find audio input device index with pattern:", pattern)


if system == "Windows":
    raise Exception("Windows is currently not supported.")
elif system == "Darwin":
    input_device_index = get_input_device_index(input_device_list, "BlackHole {}")
elif system == "Linux":
    input_device_index = get_input_device_index(
        input_device_list, "Loopback: PCM (hw:{},1)"
    )

# input_device_index = 2 # shall you automate this?
#
# Loopback: PCM (hw:{},1)

print("Recording")

stream = p.open(
    format=sample_format,
    channels=channels,  # this is microphone. how to record internal audio?
    rate=fs,
    frames_per_buffer=chunk,  # this is the chunk.
    # macos: 0 for blackhole. but you must set blackhole as output.
    input_device_index=input_device_index,
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
            t.commit()

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
    print("AudioRecorder: Can't start. Redis signal is off.".upper())
