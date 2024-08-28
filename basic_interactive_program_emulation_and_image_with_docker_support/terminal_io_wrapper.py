# count for io stats
from remote_socket import SocketClient
import sys
import tty
tty.setcbreak(sys.stdin)

# import signal

# def signal_handler(signum, frame):
#     print('Received SIGINT')
#     pass

# # Set the signal handler for SIGINT (Ctrl+C)
# signal.signal(signal.SIGINT, signal_handler)

# this script is not usable, as special combos are not received at all.

# import fcntl
# import os
# import select

# # Make stdin non-blocking
# fd = sys.stdin.fileno()
# flags = fcntl.fcntl(fd, fcntl.F_GETFL)
# fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

# try:
#     while True:
#         # Use select to wait for input
#         ready_to_read, _, _ = select.select([sys.stdin], [], [], 1.0)
        
#         if ready_to_read:
#             raw_bytes = sys.stdin.buffer.read(1)  # Read one byte
#             if not raw_bytes:  # EOF
#                 break
#             print(f'Received byte: {raw_bytes}')
#         else:
#             print('No input received, continuing...')

# except KeyboardInterrupt:
#     print('Exiting...')
# finally:
#     # Reset stdin to blocking mode if necessary
#     fcntl.fcntl(fd, fcntl.F_SETFL, flags)


def main():
    # client = SocketClient()
    total_count = 0
    while True:
        user_input = sys.stdin.buffer.read(1)
        recv_len = len(user_input)
        total_count += recv_len
        # Process the raw bytes (for demonstration, we'll just print the length)
        print(f'Received {recv_len} bytes.')
        print(f"Total bytes received:", total_count)
        # client.send(user_input)

if __name__ == '__main__':
    main()