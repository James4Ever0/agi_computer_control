import random

from type_utils import *


class NaiveVocab:
    charlist = ["a"]
    startpoint = ""
    content_typeguard = enforce_str

    @classmethod
    def generate(cls):
        content = cls.startpoint
        for _ in range(random.randint(1, 10)):
            char = random.choice(cls.charlist)
            content += char
        return content

    @classmethod
    def filter(cls, content):
        content = cls.content_typeguard(content)
        result = cls.startpoint
        for char in content:
            if char in cls.charlist:
                result += char
        return result


class AsciiVocab(NaiveVocab):
    charlist = [chr(x) for x in range(256)]


class BytesVocab(NaiveVocab):
    startpoint = b""
    charlist = [bytes([x]) for x in range(256)]
    content_typeguard = enforce_bytes
