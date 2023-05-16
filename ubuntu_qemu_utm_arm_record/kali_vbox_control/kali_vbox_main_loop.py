import time
import os
import progressbar
seconds = 60*5

restore_vm = "restore_vm.sh"
stop_vm = "stop_vm.sh"
start_vm = "start_vm.sh"
mount_kali_webdav_dirs = "../kali_prepare_two_webdav_dirs.sh"


def run_bash_script(script_path):
    print("Excuting bash script: %s" % script_path)
    command = f"bash {script_path}"
    return os.system(command)

scripts = [stop_vm, mount_kali_webdav_dirs, restore_vm, start_vm]

while True:
    codes = []
    for script in scripts:
        codes.append(script)
    if any(codes):
        print("HAS ERROR CODE!")
        for iscript in scripts:

        time.sleep(1)
        continue
    # CAN YOU DO THIS WITHOUT INTERRUPTING ME?
    print("WAITING {} SECONDS...".format(seconds))
    for _ in progressbar.progressbar(range(seconds)):
        time.sleep(1)
