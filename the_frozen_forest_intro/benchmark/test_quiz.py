from quiz import Quiz
import socket

# ip tests are inconsistent, because different runs have different results.

def build_quiz(serial: str):
    quiz = Quiz(f"test_spec/json/test_{serial}.json")
    return quiz

def test_04():
    dns_common_test('04')


def test_07():
    dns_common_test('07')

def test_01():
    dns_common_test('01')

def dns_common_test(it):
    quiz = build_quiz(it)
    right_ans = socket.gethostbyname(
        "bing.com"
    )  # not always right, just our current result.
    wrong_ans = "400.400.400.400"

    # assert True == quiz.evaluate(right_ans)
    assert quiz.answer.count('.') == 3
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
    assert_value_is_correct(1 / 5, quiz.evaluate(ans_1))
    assert_value_is_correct(2 / 5, quiz.evaluate(ans_2))
    assert_value_is_correct(2 / 5, quiz.evaluate(ans_3))
    assert_value_is_correct(4 / 5, quiz.evaluate(ans_4))


def test_05():
    quiz = build_quiz("05")

    ans_0 = "Credit & Debit Card, Apple Pay\nPayPal (Automatic Renewal)\nPayPal (One-Time)\nAlipay\nCryptocurrency (Cryptomus - No Refunds)\nMail in Payment"
    ans_1 = "it is something else"
    ans_2 = ""
    assert_value_is_correct(1, quiz.evaluate(ans_0))
    val_1 = quiz.evaluate(ans_1)
    assert val_1 > 0
    assert val_1 < 1
    assert_value_is_correct(0, quiz.evaluate(ans_2))
