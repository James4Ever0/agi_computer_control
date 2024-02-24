pattern = ['0','1']

def create_pattern(length:int):
    assert length >=1
    pattern_repetition_count = length // len(pattern)
    if length%len(pattern) !=0:
        pattern_repetition_count += 1
    mpat = pattern * pattern_repetition_count
    ret = "".join(mpat[:length])
    return ret

if __name__ == "__main__":
    ret = create_pattern(10000)
    print(ret)