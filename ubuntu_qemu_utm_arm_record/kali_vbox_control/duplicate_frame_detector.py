import os

screenshot_vm = "screenshot_vm.sh"
image_path = "output.png"
exit_code = os.system(screenshot_vm)

lock_indicator_path = "lock_indicator.png"
# right now just check the login screen. if it hits the login screen we reboot.

# or you execute command to find out the state of gnome on display :0
import pyscreeze

result = pyscreeze.locate(needleImage=lock_indicator_path,haystackImage=image_path)

if result:
    print("FIND IMAGE AT:", result)
else:
    print("IMAGE NOT FOUND.")