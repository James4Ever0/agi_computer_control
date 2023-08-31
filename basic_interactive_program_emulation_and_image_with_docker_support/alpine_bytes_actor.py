from vocabulary import BytesVocab
from alpine_actor import AlpineActor, run_actor_forever
from bytes_actor import BytesActor

class AlpineBytesActor(AlpineActor, BytesActor):
    @AlpineActor.timeit
    def loop(self):
        _ = self.read()
        write_content = BytesVocab.generate()
        self.write(write_content)
        return True

if __name__ == '__main__':
    run_actor_forever(AlpineBytesActor)