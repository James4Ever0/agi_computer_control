cd /sys/bus/usb/devices
ls -1 */power/control | xargs -Iabc bash -c "echo on > abc"
