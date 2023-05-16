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
    os.system(command)


while True:
    run_bash_script(stop_vm)
    run_bash_script(mount_kali_webdav_dirs)
    run_bash_script(restore_vm)
    run_bash_script(start_vm)
    # CAN YOU DO THIS WITHOUT INTERRUPTING ME?
    print("WAITING {} SECONDS...".format(seconds))
    for _ in progressbar.progressbar(range(seconds)):
        time.sleep(1)
