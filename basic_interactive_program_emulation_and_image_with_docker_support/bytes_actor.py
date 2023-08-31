class BytesActor(NaiveActor):
    write_method = lambda proc: proc.send

if __name__ == "__main__":
    actor = NaiveActor(f"{sys.executable} naive_interactive.py")
    actor.run()