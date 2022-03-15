import rich
import re
import subprocess
import traceback
# cmd = 'vboxmanage guestcontrol "Ubuntu 16.04" --username hua --password 110110 run --timeout 100 -- /bin/loginctl'
# cmd = ['vboxmanage', 'guestcontrol', 'Ubuntu 16.04', '--username', 'hua', '--password', '110110', 'run', '--timeout', '100', '--', '/bin/loginctl','--help']


def exec_vm_if_locked(verbose=False,
                      timeout=1  # seconds.

                      ):

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
            [],  # main info.
            [
                "show-seat",
                "seat0",  # get ActiveSession
            ],
        ]
    ]

    def cmd_unlock(session): return getcmd(['unlock-session', session])

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

    keys = ['session', 'seat']
    datamap = {}
    reboot = False

    try:
        for index, cmd in enumerate(cmds):
            key = keys[index]
            output = subprocess.check_output(cmd, timeout=timeout)
            # if with error return code, it will raise exception.
            # machine not started, service not running.
            if verbose:
                print("=============OUTPUT=============")
                print(output.decode())
                print()
            dec_output = output.decode()

            mlist_output = sess_parse(dec_output)
            if verbose:
                rich.print(mlist_output)
            mdict = {e[0]: e[1:] for e in mlist_output}

            datamap[key] = mdict
        if verbose:
            print()
            rich.print(datamap)

        active_session = datamap['seat']['ActiveSession'][0]

        user_active_session = datamap['session'][active_session][1]

        if user_active_session != 'hua':
            reboot = True
        else:
            cmd = cmd_unlock(active_session)
            subprocess.call(cmd, timeout=timeout)
    except:
        traceback.print_exc()
    return reboot


if __name__ == '__main__':
    reboot = exec_vm_if_locked(verbose=True)
    print("NEED REBOOT?", reboot)
