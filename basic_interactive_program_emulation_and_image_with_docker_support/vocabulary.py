import random


class NaiveVocab:
    charlist = ["a"]

    @classmethod
    def generate(cls):
        content = ""
        for _ in range(random.randint(1, 10)):
            char = random.choice(cls.charlist)
            content += char
        return content

    @classmethod
    def filter(cls, content):
        if isinstance(content, bytes):
            content = content.decode()
        if not isinstance(content, str):
            raise Exception("Invalid content type: %s\nShould be: str" % type(content))
        result = ""
        for char in content:
            if char in cls.charlist:
                result += char
        return result


class AsciiVocab(NaiveVocab):
    charlist = [chr(x) for x in range(256)]

class 