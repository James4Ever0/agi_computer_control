from naive_actor import NaiveActor, run_naive
class BytesActor(NaiveActor):
    write_method = lambda proc: proc.send

if __name__ == "__main__":
    run_naive(BytesActor)