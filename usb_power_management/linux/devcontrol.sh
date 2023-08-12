# ref: https://www.kernel.org/doc/html/v4.16/driver-api/usb/power-management.html
cd /sys/bus/usb/devices
ls -1 */power/control | xargs -Iabc bash -c "echo on > abc"
echo -1 > /sys/module/usbcore/parameters/autosuspend/.
ls -1 */power/usb2_hardware_lpm | xargs -Iabc bash -c "echo n > abc"
# ls -1 */power/usb3_hardware_lpm_u1 | xargs -Iabc bash -c "echo disable > abc"
# ls -1 */power/usb3_hardware_lpm_u2 | xargs -Iabc bash -c "echo disable > abc"