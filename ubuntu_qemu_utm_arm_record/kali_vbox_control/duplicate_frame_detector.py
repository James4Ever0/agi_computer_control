import os

screenshot_vm = "screenshot_vm.sh"
image_path = "output.png"
exit_code = os.system(screenshot_vm)

# right now just check the login screen. if it hits the login screen we reboot.