export OPENAI_API_KEY='any'
export OPENAI_API_BASE=http://192.168.1.4/vllm/v1
export MODEL_NAME='mixtral-local'
export BETTER_EXCEPTIONS=1

python3.9 test_agent.py
rm .sw*