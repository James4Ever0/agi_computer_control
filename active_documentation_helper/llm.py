# from langchain.prompts import Prompt
# from langchain.chains import LLMChain
from contextlib import contextmanager
from langchain.llms import OpenAI
import tiktoken


def print_center(banner: str):
    print(banner.center(50, "="))


class LLM:
    """
    A class for running a Language Model Chain.
    """

    def __init__(self, prompt: str, temperature=0, gpt_4=False):
        """
        Initializes the LLM class.
        Args:
            prompt (PromptTemplate): The prompt template to use.
            temperature (int): The temperature to use for the model.
            gpt_4 (bool): Whether to use GPT-4 or Text-Davinci-003.
        Side Effects:
            Sets the class attributes.
        """
        self.prompt = prompt
        self.prompt_size = self.number_of_tokens(prompt)
        self.temperature = temperature
        self.gpt_4 = gpt_4
        self.model_name = "gpt-4" if self.gpt_4 else "text-davinci-003"
        self.max_tokens = 4097 * 2 if self.gpt_4 else 4097
        self.show_init_config()
        self.clear_llm()

    def show_init_config(self):
        print_center("init params")
        print(f"Model: {self.model_name}")
        print(f"Max Tokens: {self.max_tokens}")
        print(f"Prompt Size: {self.prompt_size}")
        print(f"Temperature: {self.temperature}")
        print_center("init config")
        print(self.prompt)

    def create_llm(self):
        self.llm = OpenAI(
            temperature=self.temperature,
            max_tokens=-1,
            model_name=self.model_name,
            disallowed_special=(),  # to suppress error when special tokens within the input text (encode special tokens as normal text)
        )

    def clear_llm(self):
        self.llm = None

    def _run(self, query: str):
        chunk_list = []
        print_center("query")
        print(query)
        print_center("response")
        _input = "\n".join([self.prompt, query])
        for chunk in self.llm.stream(input=_input):
            print(chunk, end="", flush=True)
            chunk_list.append(chunk)
        print()

        result = "".join(chunk_list)
        return result

    def run_once(self, query: str):
        """
        Runs the Language Model Chain.
        Args:
            code (str): The code to use for the chain.
            **kwargs (dict): Additional keyword arguments.
        Returns:
            str: The generated text.
        """
        self.create_llm()

        # chain = LLMChain(llm=llm, prompt=self.prompt)
        result = self._run(query)
        self.clear_llm()
        return result

    def number_of_tokens(self, text):
        """
        Counts the number of tokens in a given text.
        Args:
            text (str): The text to count tokens for.
        Returns:
            int: The number of tokens in the text.
        """
        encoding = tiktoken.encoding_for_model("gpt-4")
        return len(encoding.encode(text, disallowed_special=()))


@contextmanager
def llm_context(prompt: str, temperature=0, gpt_4=False):
    model = LLM(prompt, temperature=temperature, gpt_4=gpt_4)
    try:
        yield model
    finally:
        del model
