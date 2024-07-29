# use ssh to do the job. 

# this could be bad practice in general. you should leave no port onto that terminal environment. isolate llm server and the terminal with restricted api calls.

# but it does offer more expediency than the multi-pod approach.

# approach 1: setup a separate pod for running ssh client, bridging host port to the target pod (requiring pod ip known, sshd bind to static ip service)

# approach 2: directly run the ssh client on host machine

TIMEOUT=10
PODNAME=openinterpreter-container
PORT_FORWARD_SESSION_NAME=openinterpreter_host_port_forward

# Run the command with a timeout of 10 seconds
timeout $TIMEOUT k3s kubectl wait pod/$PODNAME --for condition=Ready

# Check the exit code of the timeout command
if [ $? -eq 124 ]; then
    echo "Command timed out. Exiting..."
else
    echo "Command completed within 10 seconds."
fi

POD_IP=$(k3s kubectl get pods $PODNAME -o json | jq -r .status.podIP)

tmux kill-session -t $PORT_FORWARD_SESSION_NAME
tmux new -s $PORT_FORWARD_SESSION_NAME -d "ssh -N -v -R localhost:9101:localhost:8101 -o StrictHostKeyChecking=no root@$POD_IP"
