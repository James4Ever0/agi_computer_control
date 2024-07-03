import subprocess
import sys
import typing
import pydantic
import re
import os
from typing import Optional
import requests
import base64


def read_and_encode_file_as_base64(filepath: str):
    with open(filepath, "rb") as f:
        content = f.read()
    ret = base64.b64encode(content).decode("utf-8")
    return ret


def image_caption_from_path(
    image_path: str, prompt: str = "Describe the image in detail: \n"
):
    imageBase64 = read_and_encode_file_as_base64(image_path)
    data = dict(imageBase64=imageBase64, query=prompt)
    ret = requests.post("http://localhost:9002/image_chat", json=data)
    assert ret.status_code == 200, f"Failed to get caption: {ret.text}"
    response = ret.json()
    response = response["response"]
    return response


def python_script_eval_trusted(script_path: str, timeout: int = 10, strip=True):

    with open(script_path, "r") as f:
        script_content = f.read()
    ret = python_eval_trusted(script_content, timeout=timeout, strip=strip)
    return ret


def docker_python_eval_trusted(python_code: str, config: dict):
    cmd_prefixes = ["docker", "run", "--rm", config["image"]]
    python_executable = config["python_binary"]
    ret = python_eval_trusted(
        python_code, cmd_prefixs=cmd_prefixes, python_executable=python_executable
    )
    return ret


def python_eval_trusted(
    python_code: str,
    timeout: int = 10,
    strip=True,
    cmd_prefixs: list[str] = [],
    python_executable: str = sys.executable,
):
    """
    Could throw arbitrary exception which must be catched.

    Untrusted code shall not be run with this function.
    """
    cmd = [*cmd_prefixs, python_executable, "-c", python_code]
    print("[*] Executing cmd:", cmd)
    print("[*] Running python code:", python_code)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    proc_code = proc.returncode
    if proc_code != 0:
        print("[-] Python code exited with non-zero exit code:", proc_code)
    ret = proc.stdout
    if strip:
        ret = ret.strip()
    if ret == "":
        raise Exception("[-] Python code exited with empty output")
    return ret


def resolve_path_relative_to_filepath(path: str, filepath: str):
    """
    Resolve path relative to filepath.
    """
    if not os.path.isabs(path):
        script_base = os.path.split(os.path.abspath(filepath))[0]
        path = os.path.join(script_base, path)
    return path


# def get_script_content_relative_to_filepath(script_path: str, filepath: str):
#     script_path = resolve_path_relative_to_filepath(script_path, filepath)
#     with open(script_path, "r") as f:
#         script_content = f.read()
#     return script_content


class QASpec(pydantic.BaseModel):
    source: str
    content_: str
    config: dict = {}
    filepath_: Optional[str] = None

    @property
    def content_filepath(self):
        ret = resolve_path_relative_to_filepath(self.content_, self.filepath_)  # type: ignore
        return ret

    @property
    def content(self):
        if self.source == "text":
            ret = self.content_
        elif self.source == "docker_python_code":
            ret = docker_python_eval_trusted(self.content_, self.config)
        elif self.source == "python_code":
            ret = python_eval_trusted(self.content_)
        elif self.source == "python_script_path":
            ret = python_script_eval_trusted(self.content_filepath)
        elif self.source == "image_path":
            # had better not to tell anything task related to model weaker than internvl
            # since that will lead to cheating, heavily impact the overall score.
            # minicpmv requires custom ollama build.
            # https://ollama.com/hhao/openbmb-minicpm-llama3-v-2_5:latest
            ret = image_caption_from_path(self.content_filepath)
        else:
            raise NotImplementedError(
                "Unsupported QASpec source type: %s" % self.source
            )

        print("[*] QASpec source:", self.source)
        print("[*] QASpec content:", ret)
        return ret


class EvalSpec(pydantic.BaseModel):
    method: str
    config: dict = {}


class QuizSpec(pydantic.BaseModel):
    question: QASpec
    answer: QASpec
    eval: EvalSpec
    # config: dict = {}

    @classmethod
    def load_from_file(cls, filename: str):
        ret = cls.parse_file(filename)
        ret.question.filepath_ = filename
        ret.answer.filepath_ = filename
        return ret


def split_word_list(word_list: list[str], split_word: str) -> list[str]:
    ret = []
    for it in word_list:
        splited_it = it.split(split_word)
        ret.extend(splited_it)
    return ret


def split_with_chars(s: str, split_chars):
    if type(split_chars) is str:
        split_chars = [split_chars]
    assert type(split_chars) == list
    assert len(split_chars) > 0
    ret = [s]
    for it in split_chars:
        ret = split_word_list(ret, it)
    return ret


def get_words_from_string(s: str, config: dict):
    pattern = r"\w+"
    ret = None
    if config:
        split_chars = config.get("split", None)
        if split_chars is not None:
            ret = split_with_chars(s, split_chars)
        else:
            pattern = config["pattern"]
            assert type(pattern) == str
    if ret is None:
        ret = re.findall(pattern, s)
    return ret


def calculate_word_match_score(answer: str, user_answer: str, config: dict):
    target_words = get_words_from_string(answer, config)
    user_words = get_words_from_string(user_answer, config)
    if len(target_words) == 0:
        raise Exception("[-] Empty target words")
    hit_words = 0
    for it in target_words:
        if it in user_words:
            hit_words += 1
    score = hit_words / len(target_words)
    return score


def calculate_semantic_similarity_score(answer: str, user_answer: str):
    if answer.strip() == "":
        return 0
    if user_answer.strip() == "":
        return 0
    response = requests.post(
        "http://localhost:9000/calculate_similarity",
        json=dict(text1=answer, text2=user_answer),
    ).json()
    score = response["similarity"]
    return score


class Quiz:
    def __init__(self, quiz_file: str):
        self.quizSpec = QuizSpec.load_from_file(quiz_file)

    @property
    def question(self):
        return self.quizSpec.question.content

    @property
    def answer(self):
        return self.quizSpec.answer.content

    @property
    def eval_method(self):
        return self.quizSpec.eval.method

    def evaluate(self, user_answer: str) -> typing.Union[float, int, bool]:
        if self.eval_method == "contains":
            has_answer = self.answer in user_answer
            return has_answer
        elif self.eval_method == "word_match":
            match_score = calculate_word_match_score(
                self.answer, user_answer, self.quizSpec.eval.config
            )
            return match_score
        elif self.eval_method == "equal":
            is_equal = self.answer == user_answer
            return is_equal
        elif self.eval_method == "semantic_similarity":
            score = calculate_semantic_similarity_score(self.answer, user_answer)
            return score
        else:
            raise NotImplementedError(
                "Unsupported quiz evaluation method: %s" % self.eval_method
            )
