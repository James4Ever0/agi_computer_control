import pyaudio
p = pyaudio.PyAudio ()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
# make speakers recordable? macos needs blackhole.

# BEFORE
# Input Device id  0  -  MacBook Air Microphone
# Input Device id  2  -  NoMachine Audio Adapter
# Input Device id  3  -  NoMachine Microphone Adapter

# AFTER
# Input Device id  0  -  BlackHole 2ch
# Input Device id  1  -  MacBook Air Microphone
# Input Device id  3  -  NoMachine Audio Adapter
# Input Device id  4  -  NoMachine Microphone Adapter

for i in range(0, numdevices):
    if (p.get_device_info_by_host_api_device_index (0, i).get ('maxInputChannels')) > 0:
        print ("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index (0, i).get ('name'))