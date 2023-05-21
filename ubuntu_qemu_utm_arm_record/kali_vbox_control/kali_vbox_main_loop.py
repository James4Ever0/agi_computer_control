import time
import os
import progressbar
seconds = 60*5

restore_vm = "restore_vm.sh"
stop_vm = "stop_vm.sh"
start_vm = "start_vm.sh"
kali_prepare_two_webdav_dirs = "../kali_prepare_two_webdav_dirs.sh"
ABORT_THRESHOLD = 15

from exec_vm_if_locked import exec_vm_if_locked

def run_bash_script(script_path):
    print("Excuting bash script: %s" % script_path)
    command = f"bash {script_path}"
    return os.system(command)


# scripts = [stop_vm, kali_prepare_two_webdav_dirs, restore_vm] # restoration will make vm start?
scripts = [stop_vm, kali_prepare_two_webdav_dirs, restore_vm, start_vm]

while True:
    os.system("rm nohup.out")
    codes = []
    for script in scripts:
        while True:
            code = run_bash_script(script)
            time.sleep(1)
            if script != start_vm:
                break
            else:
                if code != 0:
                    continue
                else:
                    break
        codes.append(code)
    print()
    for index, script in enumerate(scripts):
        print(f'{script} EXIT CODE:', codes[index])
    # if any(codes):
    #     print()
    #     print("HAS ERROR CODE!")
    #     for index, script in enumerate(scripts):
    #         print(f'{script} EXIT CODE:', codes[index])
    #     time.sleep(1)
    #     continue
    # CAN YOU DO THIS WITHOUT INTERRUPTING ME?
    print("WAITING {} SECONDS...".format(seconds))
    abort = 0
    for _ in progressbar.progressbar(range(seconds)):
        # you just don't wait...
        paths = os.listdir("/tmp/.X11-unix")
        target_path = "X10"  # where virtualbox runs.
        vm_running = target_path in paths
        if not vm_running:
            abort += 1
        else:
            reboot = exec_vm_if_locked()
            if reboot:
                abort +=1
            else:
                abort = 0
        if abort > ABORT_THRESHOLD:
            print("ABORTING! VM IS NOT RUNNING.")
            break
        time.sleep(1)
    print()
