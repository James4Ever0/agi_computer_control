import subprocess
import sys
import typing
import pydantic
import re


def python_eval_trusted(python_code: str, timeout: int = 10, strip=True):
    """
    Could throw arbitrary exception which must be catched.

    Untrusted code shall not be run with this function.
    """
    cmd = [sys.executable, "-c", python_code]
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


class QASpec(pydantic.BaseModel):
    source: str
    content_: str

    @property
    def content(self):
        if self.source == "text":
            ret = self.content_
        elif self.source == "python_code":
            ret = python_eval_trusted(self.content_)
        else:
            raise NotImplementedError(
                "Unsupported QASpec source type: %s" % self.source
            )
        print("[*] QASpec source:", self.source)
        print("[*] QASpec content:", ret)
        return ret


class EvalSpec(pydantic.BaseModel):
    method: str


class QuizSpec(pydantic.BaseModel):
    question: QASpec
    answer: QASpec
    eval: EvalSpec


def get_words_from_string(s: str):
    return re.findall(r"\w+", s)


def calculate_word_match_score(answer: str, user_answer: str):
    target_words = get_words_from_string(answer)
    user_words = get_words_from_string(user_answer)
    if len(target_words) == 0:
        raise Exception("[-] Empty target words")
    hit_words = 0
    for it in target_words:
        if it in user_words:
            hit_words += 1
    score = hit_words / len(target_words)
    return score


class Quiz:
    def __init__(self, quiz_file: str):
        self.quizSpec = QuizSpec.parse_file(quiz_file)

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
            match_score = calculate_word_match_score(self.answer, user_answer)
            return match_score
        elif self.eval_method == "equal":
            is_equal = self.answer == user_answer
            return is_equal
        else:
            raise NotImplementedError("Unsupported eval method: %s" % self.eval_method)
