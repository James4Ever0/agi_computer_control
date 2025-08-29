# TODO: store the cookie in home directory, use environment variable COOKIE_STORAGE_LOCATION
# echo "Removing client cookie"
# rm /tmp/.ctf-client-cookie

# Register a user
echo "Registering a user"
python client.py register username

# View whoami
echo "Viewing whoami"
python client.py whoami

# TODO: implement and test client score, info
# TODO: implement a file server directly in server, mount the folder according to the challenge.yaml (must not be exploited by dir traversal)
# TODO: implement file download command using client

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