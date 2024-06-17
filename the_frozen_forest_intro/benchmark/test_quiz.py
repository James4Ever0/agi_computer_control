from quiz import Quiz
import socket

def build_quiz(serial:str):
    quiz = Quiz(f"test_spec/json/test_{serial}.json")
    return quiz

def test_01_04():
    candidates = ['01', '04']
    for it in candidates:
        quiz = build_quiz(it)
        right_ans = socket.gethostbyname(
            "bing.com"
        )  # not always right, just our current result.
        wrong_ans = "400.400.400.400"

        assert True == quiz.evaluate(right_ans)
        assert False == quiz.evaluate(wrong_ans)
        
def assert_value_is_correct(value, correct_value, eps=1e-4):
    assert abs(value - correct_value) <= eps

def test_02():
    quiz = build_quiz("02")
    
    ans_0 = "i do not have nothing for you"
    ans_1 = "80 443"
    ans_2 = "400\n2000 5060"
    ans_3 = "111,2000"
    ans_4 = "22, 80, 111, 2000"
    
    assert_value_is_correct(0, quiz.evaluate(ans_0))
    assert_value_is_correct(1/5, quiz.evaluate(ans_1))
    assert_value_is_correct(2/5, quiz.evaluate(ans_2))
    assert_value_is_correct(2/5, quiz.evaluate(ans_3))
    assert_value_is_correct(4/5, quiz.evaluate(ans_4))