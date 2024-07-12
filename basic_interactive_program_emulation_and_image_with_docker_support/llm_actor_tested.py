# fallback to any server usable, use a tested server backend.

# will use a dedicated local server for the fallback implementation, doing regular server alive or not checks

import requests
from naive_actor_tested import prepare_and_eval_quiz_env

LLM_PORT = 8540
LLM_FALLBACK_SERVER_API_URL = f"http://localhost:{LLM_PORT}/llm_chat"

def llm_action_generator(observation:str):
    query = f"""Observation: {observation}
    
    Target: Get IP address of bing.com

    Instruction: You will type comamnd into a terminal and get the answer.
    """.lstrip()
    ans = requests.get(LLM_FALLBACK_SERVER_API_URL, params = dict(query=query))
    action= ans.text
    return action

def main():
    prepare_and_eval_quiz_env(llm_action_generator)

if __name__ == "__main__":
    main()
