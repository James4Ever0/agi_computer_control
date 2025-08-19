import pexpect
process = pexpect.spawn('bash', dimensions=(80,25))
process.read(1024)
input_bytes = b'ls\n'
process.send(input_bytes)