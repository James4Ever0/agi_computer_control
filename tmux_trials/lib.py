from typing import Optional, Iterable
import wcwidth
import subprocess
import os
import traceback
import shutil

TMUX = "tmux"
TMUX_WIDTH = 80
TMUX_HEIGHT = 24

REQUIRED_BINARIES = (TMUX,)


def assert_binary_existance(binaries: Iterable[str]):
    for it in binaries:
        assert shutil.which(it) != None, f"Binary '{it}' not found in path"


assert_binary_existance(REQUIRED_BINARIES)


class TmuxServer:
    def __init__(self, name: Optional[str] = None, reset: bool = True):
        if name is None:
            print("[*] Falling back to default server name")
            name = "default"
        self.name = name
        self.prefix = f"{TMUX} -L {self.name}"
        self.prefix_list = self.prefix.split(" ")
        print(f"[*] Tmux server using name '{self.name}'")

        if reset:
            self.reset()

    def reset(self):
        exit_code = self.execute_command("kill-server")
        ret = exit_code == 0
        if ret:
            print(f"[+] Server '{self.name}' reset complete")
        else:
            print(f"[+] Server '{self.name}' reset failed with exit code", exit_code)
        return ret

    def create_session(self, name: str, command: str):
        try:
            ret = TmuxSession(name, self, command)
            print(f"[+] Tmux session '{name}' created")
            return ret
        except TmuxSessionCreationFailure:
            traceback.print_exc()
            print(f"[-] Failed to create tmux session named '{name}'")

    def set_session_option(self, name: str, key: str, value: str):
        cmd = self.prepare_command(f"set-option -t {name} {key} {value}")
        self.execute_command(cmd)

    def kill_session(self, name: str):
        cmd = self.prepare_command(f"kill-session -t {name}")
        self.execute_command(cmd)

    def create_env(self, name: str, command: str):
        session = self.create_session(name, command)
        if session:
            ret = TmuxEnv(session)
            print("[+] Tmux env created")
            return ret
        else:
            print("[-] Failed to create tmux env")

    def prepare_command(self, suffix: str):
        ret = f"{self.prefix} {suffix}"
        return ret

    def prepare_command_list(self, suffix_list: list[str]):
        ret = self.prefix_list + suffix_list
        return ret

    def get_command_output_bytes(self, suffix: str):
        cmd = self.prepare_command(suffix)
        ret = subprocess.check_output(cmd)
        return ret

    def execute_command(self, suffix: str):
        cmd = self.prepare_command(suffix)
        ret = os.system(cmd)
        return ret

    def execute_command_list(self, suffix_list: list[str]):
        cmd_list = self.prepare_command_list(suffix_list)
        ret = subprocess.Popen(cmd_list).wait()
        return ret


class TmuxSession:
    def __init__(
        self,
        name: str,
        server: TmuxServer,
        command: str,
        width=TMUX_WIDTH,
        height=TMUX_HEIGHT,
        isolate=True,
        kill_existing=True,
    ):
        if kill_existing:
            print(f"[*] Killing session '{name}' before creation")
            server.kill_session(name)
        exit_code = server.execute_command(
            f"new-session -d -s {name} -x {width} -y {height} {command}"
        )
        if exit_code != 0:
            raise TmuxSessionCreationFailure(
                f"Tmux session creation command exited with code {exit_code}"
            )
        if isolate:
            print(f"[*] Performing isolation for tmux session '{name}")
            self.isolate()
        self.name = name
        self.server = server

    def isolate(self):
        self.set_option("prefix", "None")
        self.set_option("prefix2", "None")
        self.set_option("status", "off")
        self.set_option("aggressive-resize", "off")
        self.set_option("window-size", "manual")

    def preview_bytes(self):
        ret = self.server.get_command_output_bytes(f"capture-pane -t {self.name} -p")
        return ret

    def kill(self):
        self.server.kill_session(self.name)
        del self

    def set_option(self, key: str, value: str):
        self.server.set_session_option(self.name, key, value)
    
    def info(self):
        ...


class TmuxEnv:
    def __init__(self, session: TmuxSession):
        self.session = session
        self.server = session.server

    def send_keys(self, keys: list[str]):
        command_list = ["send-keys", "-t", self.session.name, *keys]
        self.server.execute_command_list(command_list)
    
    def info(self):
        ret = self.session.info()
        return ret

    def kill(self):
        self.session.kill()
        del self


class TmuxSessionCreationFailure(Exception): ...
