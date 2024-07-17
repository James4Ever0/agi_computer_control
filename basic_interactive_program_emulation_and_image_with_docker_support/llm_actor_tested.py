# fallback to any server usable, use a tested server backend.

# will use a dedicated local server for the fallback implementation, doing regular server alive or not checks

import requests
from naive_actor_tested import prepare_and_eval_quiz_env
import markdown
from bs4 import BeautifulSoup

LLM_PORT = 8540
LLM_FALLBACK_SERVER_API_URL = f"http://localhost:{LLM_PORT}/llm_chat"


def extract_commands(response: str):
    commands = []
    response_html = markdown.markdown(response)
    soup = BeautifulSoup(response_html, features="lxml")
    for it in soup.find_all("code"):
        commands.append(it.text)
    # commands = re.findall(r"`([^`]+)`",response)
    return commands


def llm_action_generator(observation: str):
    query = "How to get IP address of bing.com in bash terminal?"  # so try to make everything conversational?
    # the answer is just not so parser friendly
    response = requests.get(LLM_FALLBACK_SERVER_API_URL, params=dict(query=query))
    llm_response = response.text
    print("[*] LLM Response:")
    print(llm_response)
    # action = llm_response
    action = "\n".join(extract_commands(llm_response))
    return action


def main():
    prepare_and_eval_quiz_env(llm_action_generator)


if __name__ == "__main__":
    main()
