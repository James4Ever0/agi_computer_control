# conda run -n cybergod --no-capture-output interpreter -m mixtral-local -v -t 0.1 -ab http://localhost:8101/v1
# conda run -n cybergod --no-capture-output python open_interpreter_tested.py
CONTAINER_NAME=openinterpreter_container
SCRIPT_NAME=open_interpreter_tested.py

PROMPT_PATH=/tmp/prompt.txt

docker kill $CONTAINER_NAME
docker rm $CONTAINER_NAME

docker run -d --rm --name $CONTAINER_NAME -e OPENAI_API_BASE=http://localhost:8101/v1 --network host openinterpreter tail -f /dev/null

docker cp $SCRIPT_NAME $CONTAINER_NAME:/root/$SCRIPT_NAME
docker cp $PROMPT_PATH $CONTAINER_NAME:$PROMPT_PATH

docker exec -it $CONTAINER_NAME python3 /root/$SCRIPT_NAME

# the bloody agent will get stuck into "which sudo"

# a proper terminal execution cycle would be:
# 1. generate a long sequence beforehand 
# 2. constantly monitor changes and track execution progress
# 3. ask the agent to continue execution or not
# 4. allow the agent to change the execution sequence on the fly

# things can either be right or wrong, but can you always make it both right and wrong the same time?