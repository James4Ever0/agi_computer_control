import random

def ensure_str(content):
    if isinstance(content, bytes):
    content = content.decode()
    if not isinstance(content, str):
        raise Exception("Invalid content type: %s\nShould be: str" % type(content))
    return content
class NaiveVocab:
    charlist = ["a"]
    startpoint = ""

    @classmethod
    def generate(cls):
        content = cls.startpoint
        for _ in range(random.randint(1, 10)):
            char = random.choice(cls.charlist)
            content += char
        return content

    @classmethod
    def filter(cls, content):
        content = 
        result = cls.startpoint
        for char in content:
            if char in cls.charlist:
                result += char
        return result


class AsciiVocab(NaiveVocab):
    charlist = [chr(x) for x in range(256)]

class BytesVocab(NaiveVocab):
    startpoint = b""
    charlist = [bytes(x) for x in range(256)]