
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--port", type=int, default=8788, help="port number")
args = parser.parse_args()
port = args.port
assert port > 0 and port < 65535