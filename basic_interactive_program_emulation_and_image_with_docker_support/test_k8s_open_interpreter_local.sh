# conda run -n cybergod --no-capture-output interpreter -m mixtral-local -v -t 0.1 -ab http://localhost:8101/v1
# conda run -n cybergod --no-capture-output python open_interpreter_tested.py
CONTAINER_NAME=openinterpreter-container
SCRIPT_NAME=open_interpreter_tested.py

WAIT_TIMEOUT=15

PROMPT_PATH=/tmp/prompt.txt

K8S_PREFIX="k3s kubectl"

$K8S_PREFIX delete pod $CONTAINER_NAME

$K8S_PREFIX apply -f no-intranet-network-policy.yaml

$K8S_PREFIX run $CONTAINER_NAME --env="OPENAI_API_BASE=http://10.42.0.14:8101/v1" --overrides='{"metadata": {"labels": {"network": "no-intranet"}}, "spec": {"dnsPolicy": "None", "dnsConfig": {"nameservers": ["8.8.8.8"]}}}' --image=openinterpreter:latest --image-pull-policy="Never" -- tail -f /dev/null

echo "waiting pod to be ready (timeout: $WAIT_TIMEOUT secs)"

timeout $WAIT_TIMEOUT $K8S_PREFIX wait pod/$CONTAINER_NAME --for condition=Running

# $K8S_PREFIX port-forward pod/$CONTAINER_NAME 8101

$K8S_PREFIX cp $SCRIPT_NAME $CONTAINER_NAME:/root/$SCRIPT_NAME
$K8S_PREFIX cp $PROMPT_PATH $CONTAINER_NAME:$PROMPT_PATH

$K8S_PREFIX exec -it $CONTAINER_NAME -- python3 /root/$SCRIPT_NAME

# the bloody agent will get stuck into "which sudo"

# a proper terminal execution cycle would be:
# 1. generate a long sequence beforehand 
# 2. constantly monitor changes and track execution progress
# 3. ask the agent to continue execution or not
# 4. allow the agent to change the execution sequence on the fly

# things can either be right or wrong, but can you always make it both right and wrong the same time?