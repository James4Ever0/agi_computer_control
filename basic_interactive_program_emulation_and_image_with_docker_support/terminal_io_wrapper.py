# count for io stats
from remote_socket import SocketClient
import pexpect
import time
import threading
import numpy as np

TMUX_SESSION_NAME='test_tmux'

def read_bytes_from_proc_and_update_stats(proc:pexpect.spawn, stats:dict):
    count = 0
    while proc.isalive():
        one_byte = proc.read(1)
        count += 1
        stats['count'] = count
        stats['bytes'] += one_byte
        # print("[*] Received one byte at time:", time.time())
        # print('[*] Total bytes received:', count)

def calculate_nth_average_speed(datalist:list, nth:int):
    nth = min(len(datalist), nth)
    diff = np.diff(datalist)
    ret = sum(diff[:nth]) / nth
    ret = float(-ret)
    return ret

def main():
    cli = f"tmux attach -t {TMUX_SESSION_NAME}"
    proc = pexpect.spawn(cli, timeout=None)
    stats = {'count':0, 'bytes': b''}

    read_thread = threading.Thread(target=read_bytes_from_proc_and_update_stats, args=(proc, stats), daemon=True)
    read_thread.start()


    datapoints = []

    maxpoints = 10
    terminal_idle = True
    while proc.isalive():
        time.sleep(1)
        # print("-"*50)
        # print('[*] Current time:', time.time())
        # print('[*] Current IO stats:', stats)
        count = stats.get('count')
        read_bytes = stats.get('bytes')
        stats['bytes'] = b'' # clear the clutter
        datapoints.insert(0, count)
        if len(datapoints) > maxpoints:
            datapoints = datapoints[:maxpoints+1]
        ten_seconds_avg = calculate_nth_average_speed(datapoints, 10)
        five_seconds_avg = calculate_nth_average_speed(datapoints,5)
        three_seconds_avg = calculate_nth_average_speed(datapoints,3)
        one_second_avg = calculate_nth_average_speed(datapoints, 1)
        # print('[*] Average bytes received in the last 10 seconds:', ten_seconds_avg)
        # print('[*] Average bytes received in the last five seconds:', five_seconds_avg)
        # print('[*] Average bytes received in the last three seconds:', three_seconds_avg)
        # print('[*] Average bytes received in the last second:', one_second_avg)
        if ten_seconds_avg > 0:
        # if five_seconds_avg > 0:
            if terminal_idle:
                terminal_idle = False
                print('[*] TerminalActive event fired!')
        else:
            if not terminal_idle:
                terminal_idle = True
                print('[*] TerminalIdle event fired!')
    print('[*] Process exited with status code:', proc.status)

if __name__ == '__main__':
    main()