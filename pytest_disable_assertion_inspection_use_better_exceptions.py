# commandline:
# env BETTER_EXCEPTIONS=1 pytest --assert=plain 

def test_mytest():
    a = 1
    b = {}
    print(b[1])
    assert b[2] == a