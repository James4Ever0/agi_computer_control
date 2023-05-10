parec --monitor-stream="$(pacmd list-sink-inputs | awk '$1 == "index:" {print $2}')" | opusenc --raw - $(xdg-user-dir MUSIC)/recording-$(date +"%F_%H-%M-%S").opus
