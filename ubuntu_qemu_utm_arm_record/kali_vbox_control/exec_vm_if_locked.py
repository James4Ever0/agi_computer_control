import subprocess

# cmd = 'vboxmanage guestcontrol "Ubuntu 16.04" --username hua --password 110110 run --timeout 100 -- /bin/loginctl'
# cmd = ['vboxmanage', 'guestcontrol', 'Ubuntu 16.04', '--username', 'hua', '--password', '110110', 'run', '--timeout', '100', '--', '/bin/loginctl','--help']
cmd = ['vboxmanage', 'guestcontrol', 'Ubuntu 16.04', '--username', 'hua', '--password', '110110', 'run', '--timeout', '100', '--', '/bin/loginctl' ,, '--output=json',''list-sessions'']

timeout = 1 # seconds.
output = subprocess.check_output(cmd, timeout=timeout)
# if with error return code, it will raise exception.
# machine not started, service not running.
print('=============OUTPUT=============')
print(output.decode())