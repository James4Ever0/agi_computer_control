# from remote_socket import SocketServer
from naive_actor import NaiveActor, run_naive
from contextlib import contextmanager

# TODO: make sure the terminal width and height is bigger than 80x25, for example, 100x30

class RelayActor(NaiveActor):
    # def recv_write_bytes(self):
    #     ret = ...
    #     return ret
    # def write_loop(self):
    #     write_bytes = self.recv_write_bytes()
    #     self.write(write_bytes)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mute = True

    def process_alive(self):
        ret = self.process.isalive()
        return ret
    
    def log_read_bytes(self, content):
        if content:
            super().log_read_bytes(content)

    @contextmanager
    def unmute(self):
        try:
            self.mute = False
            yield
        finally:
            self.mute = True
    
    def heartbeat(self):
        ret = False
        if self.process_alive():
            ret = super().heartbeat()
        return ret
    
    def loop(self):
        with self.unmute():
            self.read()
        # self.write_loop()
        return True


if __name__ == "__main__":
    run_naive(RelayActor)
