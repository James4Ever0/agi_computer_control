import os

screenshot_vm = "screenshot_vm.sh"
image_path = "output.png"
exit_code = os.system(screenshot_vm)

lock_indicator_path = "lock_indicator.png"
# right now just check the login screen. if it hits the login screen we reboot.