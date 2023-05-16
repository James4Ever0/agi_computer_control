import time
import os
restore_vm = "restore_vm.sh"
stop_vm = "stop_vm.sh"
start_vm = "start_vm.sh"
mount_kali_webdav_dirs = "../mount_kali_webdav_dirs.sh"

def run_bash_script(script_path):
    os.system("")
while True:
    os.system(stop_vm)
    os.system()
    os.system(restore_vm)
    os.system(start_vm)