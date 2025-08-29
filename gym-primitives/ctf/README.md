CTF Server and Client

This directory contains the code for the Capture The Flag (CTF) server and client. The server is responsible for managing the game state and providing an API for the client to interact with. The client is responsible for connecting to the server, receiving game state updates, and sending actions to the server.

The server is running on host 0.0.0.0 and port 8000 by default.

The client is connected to the server at 127.0.0.1:8000 by default.

SSL is optional.

First install all requirements:

```bash
pip install -r requirements.txt
```

To generate SSL certificates, run the following commands:

```bash
# this will generate the following files:
# - key.pem (private key)
# - cert.pem (certificate)
bash generate_ssl_keys.sh
```

To start the server, run the following command:

```bash
# if you want to use ssl, export those variables
export USE_SSL=1
export SSL_CERT=cert.pem
export SSL_KEY=key.pem

# run the server
python3 server_persistant.py
```

To use the client, run the following command:

```bash
# if you want to use ssl, export those variables
export SSL_CERT=cert.pem

# Register a user
echo "Registering a user"
python client.py register username

# View whoami
echo "Viewing whoami"
python client.py whoami

# View score
echo "Viewing score"
python client.py score

# View info
echo "Viewing info"
python client.py info

# List challenges
echo "Listing challenges"
python client.py list

# List all challenges
echo "Listing all challenges"
python client.py list_all

# Get a challenge
echo "Getting a challenge"
python client.py get challenge1

# Submit a solution
echo "Submitting a solution to challenge1"
python client.py submit challenge1 'cybergod{123...}'


echo "Submitting a solution to challenge2"
python client.py submit challenge2 'cybergod{123...}'

# Submit a real solution
echo "Submitting a real solution"
python client.py submit challenge1 'cybergod{12345678-1234-5678-9012-345678901234}'
# View scoreboard
echo "Viewing scoreboard"
python client.py scoreboard

# Download file
echo "Downloading a file"
python client.py download example_file.txt

# View submission limit
echo "Viewing submission limit"
python client.py submission_limit
```

The challenge collection file template:

```yaml
name: "Capture The Flag"
description: "A Capture The Flag challenge"
solution_prefix: "cybergod{"
solution_suffix: "}"
solution_max_length: 46
solution_min_length: 11
challenge_min_points: 1
challenge_max_points: 30
solution_format_description: "The solution is a variable length flag in the format cybergod{xxxx}"
file_loadpath: "static"
max_submissions: 1000
challenges:
  - id: "challenge1"
    title: "First Challenge"
    description: "This is the first challenge. Find the flag!"
    solution: "cybergod{12345678-1234-5678-9012-345678901234}"
    points: 10
    unlocks: ["challenge2"]
  - id: "challenge2"
    title: "Second Challenge"
    description: "This is the second challenge. Find the hidden flag!"
    solution: "cybergod{87654321-4321-8765-2109-876543210987}"
    points: 20
    locked: true
```