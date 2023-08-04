import mss
import timeit


def test_screenshot():
    with mss.mss() as m:
        img = m.grab(m.monitors[0])
        print(img)


output = timeit.timeit(test_screenshot, number=10)
print(output)  # 0.03 seconds
