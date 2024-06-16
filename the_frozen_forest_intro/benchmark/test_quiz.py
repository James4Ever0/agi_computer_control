from quiz import Quiz

def test_01():
    quiz = Quiz("test_01.json")
    import socket

    right_ans = socket.gethostbyname(
        "bing.com"
    )  # not always right, just our current result.
    wrong_ans = "400.400.400.400"

    assert True == quiz.evaluate(right_ans)
    assert False == quiz.evaluate(wrong_ans)

def test_02():
    quiz = Quiz("test_02.json")