# import time
from vocabulary import NaiveVocab

# will sleep for random time and respond.


class NaiveInteractive:
    def __init__(self, intro: str = "Input something and this program will reply:"):
        self.sleep = 1
        self.intro = intro
        print(self.intro)

    def loop(self):
        input()
        print(NaiveVocab.generate())
        return True

    def run(self):
        while self.loop():
            ...


if __name__ == "__main__":
    interactive = NaiveInteractive()
    interactive.run()
