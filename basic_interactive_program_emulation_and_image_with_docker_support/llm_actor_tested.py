# fallback to any server usable, use a tested server backend.

# will use a dedicated local server for the fallback implementation, doing regular server alive or not checks

import requests
from naive_actor_tested import prepare_quiz_env, eval_quiz_env

# from naive_actor_tested import prepare_and_eval_quiz_env
# import markdown
import marko
from bs4 import BeautifulSoup
import functools

LLM_PORT = 8540
LLM_FALLBACK_SERVER_API_URL = f"http://localhost:{LLM_PORT}/llm_chat"


def extract_commands(response: str):
    commands = []
    # response_html = markdown.markdown(response)
    response_html = marko.convert(response)
    soup = BeautifulSoup(response_html, features="lxml")
    print("[*] Soup:")
    print(soup)
    for pre in soup.find_all("pre"):
        for code in pre.find_all("code"):
            language = code.get("class", None)
            if language:
                language = "".join(language)
                language = language.split("language-")[-1]
            else:
                language = "unspecified"
            print("[*] Language:", language)
            code_content = code.text
            print("[*] Code:", code_content)
            commands.append(code_content)
    # commands = re.findall(r"`([^`]+)`",response)
    return commands


def llm_action_generator(observation: str, question: str):
    system = """
    
    [[SYSTEM]]
    
    You are a professional interpreter, capable of doing anything.
    
    To execute code you need to write your code within triple backticks like:

    ```language
    code
    ```

    """.lstrip()
    query = f"""{system}

    [[USER]]
    
    {question}

    [[OBSERVATION]]

    {observation}

    """.lstrip()  # so try to make everything conversational?
    # the answer is just not so parser friendly
    response = requests.get(
        LLM_FALLBACK_SERVER_API_URL, params=dict(query=query, system=system)
    )
    llm_response = response.text
    print("[*] LLM Response:")
    print(llm_response)
    # action = llm_response
    action = "\n".join(extract_commands(llm_response))
    return action


def main(num=7):
    print("[*] Quiz num:", num)
    quizEnv = prepare_quiz_env(num)
    question = quizEnv["quiz"].question
    print("[*] Quiz question:", question)
    action_generator = functools.partial(llm_action_generator, question=question)
    eval_quiz_env(quizEnv, action_generator)
    # prepare_and_eval_quiz_env(llm_action_generator)


if __name__ == "__main__":
    main()
