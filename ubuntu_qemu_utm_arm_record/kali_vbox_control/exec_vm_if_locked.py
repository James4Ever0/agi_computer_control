import rich
import re
import subprocess

# cmd = 'vboxmanage guestcontrol "Ubuntu 16.04" --username hua --password 110110 run --timeout 100 -- /bin/loginctl'
# cmd = ['vboxmanage', 'guestcontrol', 'Ubuntu 16.04', '--username', 'hua', '--password', '110110', 'run', '--timeout', '100', '--', '/bin/loginctl','--help']


def getcmd(
    args: list[str] = [],
    machine="Ubuntu 16.04",
    username="hua",
    password="110110",
    timeout="100",
    bin="/bin/loginctl",
):
    cmd = [
        "vboxmanage",
        "guestcontrol",
        machine,
        "--username",
        username,
        "--password",
        password,
        "run",
        "--timeout",
        timeout,
        "--",
        bin,
    ]
    return cmd + args


cmds = [
    getcmd(e)
    for e in [
        [], # main info.
        [
            "show-seat",
            "seat0",  # get ActiveSession
        ],
    ['',''] 
    ]
]


def sess_parse(data):
    lines = data.split("\n")
    mlist = []
    for l in lines:
        if "USER" in l:
            continue
        else:
            l = l.replace("=", " ")
            list_l = re.findall(r"\w+", l)
            # print(list_l)
            if list_l == []:
                break
            mlist.append(list_l)
    return mlist


# json is for journal formatting.


timeout = 1  # seconds.
for cmd in cmds:
    output = subprocess.check_output(cmd, timeout=timeout)
    # if with error return code, it will raise exception.
    # machine not started, service not running.
    print("=============OUTPUT=============")
    print(output.decode())
    print()
    dec_output = output.decode()

    mlist_output = sess_parse(dec_output)
    rich.print(mlist_output)
