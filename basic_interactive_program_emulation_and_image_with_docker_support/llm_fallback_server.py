import fastapi
import requests
import threading
from fastapi.responses import PlainTextResponse
from abc import ABC, abstractmethod
import subprocess
import traceback
import time

CHECK_ALIVE_INTERVAL = 5

class ServerList:
    def __init__(self):
        self.items:list["LLMChat"] = []
        self.check_alive_thread = threading.Thread(target=self.check_alive_thread_task, daemon=True)
        self.check_alive_thread.start()
    
    def check_alive_once(self):
        for it in self.items:
            try:
                it.test()
            except:
                traceback.print_exc()
                print("[-] Error checking alive:", it.__class__.__name__, it.config)
    
    def check_alive_thread_task(self):
        while True:
            self.check_alive_once()
            time.sleep(CHECK_ALIVE_INTERVAL)

    def append(self, it:"LLMChat"):
        self.items.append(it)

    def __iter__(self):
        for it in self.items:
            yield it

LOCAL_FALLBACK_OLLAMA_MODEL = "qwen2:0.5b"
LOCAL_OLLAMA_PORT = 11434
LOCAL_OLLAMA_ENDPOINT = f"http://localhost:{LOCAL_OLLAMA_PORT}"

VLLM_PORT = 8101
VLLM_OPENAI_ENDPOINT = f"http://localhost:{VLLM_PORT}"

app = fastapi.FastAPI()

ONLINE_LLM_SERVICES: ServerList = ServerList()

class LLMChat(ABC):
    def __init__(self, config: dict = {}):
        self.config = config
        self.alive = False
        self.init_attributes()

    @abstractmethod
    def init_attributes(self):
        pass

    @abstractmethod
    def test_method(self) -> bool:
        pass

    @abstractmethod
    def request_method(self, query: str) -> str:
        pass

    def test(self) -> bool:
        alive = self.test_method()
        self.alive = alive
        return alive

    def request(self, query: str):
        response = self.request_method(query)
        return response


class LocalOllamaChat(LLMChat):
    def init_attributes(self):
        self.api_endpoint = self.config["api_endpoint"]

    def test_method(self):
        test_command = "ollama list".split()
        success = False
        try:
            subprocess.check_call(test_command)
            success = True
        except subprocess.CalledProcessError:
            print("[-] Ollama not running")
        return success

    def request_method(self, query: str):
        chat_completion_endpoint = f"{self.api_endpoint}/api/generate"
        data = {"model": LOCAL_FALLBACK_OLLAMA_MODEL, "prompt": query, "stream": False}
        response = requests.post(chat_completion_endpoint, json=data)
        ret = response.json()["response"]
        return ret


class VllmOpenAIChat(LLMChat):
    def init_attributes(self):
        self.api_endpoint = self.config["api_endpoint"]
        self.model_name = self.config["model_name"]
        self.system_prompt = self.config.get('system_prompt', "You are a helpful assistant.")

    def test_method(self):
        response = requests.get(f"{VLLM_OPENAI_ENDPOINT}/")
        return response.json()["detail"] == "Not Found"

    def request_method(self, query: str):
        response = requests.post(
            f"{VLLM_OPENAI_ENDPOINT}/v1/chat/completions",
            json={
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query},
                ],
                "stream": False
            },
        )
        ret = response.json()['message']['content']
        return ret


localOllamaChat = LocalOllamaChat(config=dict(api_endpoint=LOCAL_OLLAMA_ENDPOINT))

vllmOpenAIChat = VllmOpenAIChat(
    config=dict(api_endpoint=VLLM_OPENAI_ENDPOINT, model_name="mixtral-local")
)

ONLINE_LLM_SERVICES.append(vllmOpenAIChat)

def processQueryWithOnlineLLMServices(query: str):
    try:
        for it in ONLINE_LLM_SERVICES:
            if it.alive:
                llm_response = it.request(query)
                return llm_response
            else:
                print("[-] Not alive:", it.__class__.__name__, it.config)
    except:
        traceback.print_exc()
        print("[-] Failed to use online LLM services")


def processQueryWithLocalOllamaService(query: str):
    if localOllamaChat.test():
        llm_response = localOllamaChat.request(query)
        return llm_response


@app.get("/llm_chat")
def llm_chat_with_fallback(query: str):
    # try once if the latest "available" llm has reply
    llm_response = processQueryWithOnlineLLMServices(query)
    # if not, force to use local ollama llm
    if llm_response is None:
        print("[*] Fallback to local ollama service")
        llm_response = processQueryWithLocalOllamaService(query)
    # and if not, just throw the exception.
    llm_response_type = type(llm_response)
    assert (
        type(llm_response) == str
    ), f"LLM response is not string, but {llm_response_type}"
    return PlainTextResponse(content = llm_response)
