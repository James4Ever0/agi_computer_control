from scipy.stats import entropy
import numpy as np


from contextlib import contextmanager


class EntropyCalculator:
    def __init__(self, base=2):
        self.cats_to_count = {}
        self.base = base

    def count(self, elem):
        self._count(elem)

    def _count(self, elem):
        self.cats_to_count[elem] = self.cats_to_count.get(elem, 0) + 1

    @property
    def entropy(self):
        vals = list(self.cats_to_count.values())
        if vals != []:
            hist = np.array(vals)
            total_count = sum(hist)
            hist = hist / total_count
        else:
            hist = []
        ent = entropy(hist, base=self.base)
        return ent


class ContentEntropyCalculator(EntropyCalculator):
    def count(self, content):
        if isinstance(content, str):
            content = content.encode()
        if not isinstance(content, bytes):
            raise Exception("unknown content type:", type(content))
        content_int_arr = list(content)
        for i in content_int_arr:
            self._count(i)


@contextmanager
def entropyContext(is_content=False):
    if is_content:
        calc = ContentEntropyCalculator()
    else:
        calc = EntropyCalculator()
    try:
        yield calc
    finally:
        del calc


def calculate_content_entropy(content):
    with entropyContext(is_content=True) as calc:
        calc.count(content)
        return calc.entropy

# def calculate_content_entropy(content, full=False):
# if isinstance(content, str):
#     content = content.encode()
# if not isinstance(content,bytes):
#     raise Exception("unknown content type:", type(content))
# content_int_arr = list(content)
# if full:
#     # use 256-1 bins
#     hist, _ = np.histogram(content_int_arr, bins=255, range=(0,255))
# else:
#     cats = list(set(content_int_arr))
#     cat_to_index = {cat:i for i, cat in enumerate(cats)}
#     hist = np.zeros(len(cats))
#     for elem in content_int_arr:
#         index = cat_to_index[elem]
#         hist[index] += 1
# # normalize histogram.
# hist = hist.astype(float)
# norm_hist = hist/len(content_int_arr)
# # print('normalized histogram:', norm_hist)
# ent = entropy(norm_hist, base=2)
# return ent


if __name__ == "__main__":
    testcases = ["aa", "ab", "abcdesd", "def", "hijklmn", bytes(range(256))]

    # the more chars the more entropy.

    for case in testcases:
        ent = calculate_content_entropy(case)
        # ent_full = calculate_content_entropy(case, full=True)
        print("testcase:", case)
        # identical!
        print("entropy:", ent)
        # print("local entropy:", ent)
        # print("global entropy:", ent_full)
        print()
