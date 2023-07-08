# commandline:
# env BETTER_EXCEPTIONS=1 pytest --assert=plain pytest_disable_assertion_inspection_use_better_exceptions.py

def test_mytest():
    a = 1
    b = {}
    print(b[1])
    assert b[2] == a

if __name__ == "__main__":
    test_mytest()