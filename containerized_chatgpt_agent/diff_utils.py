import difflib

join_result = lambda result: "\n".join(result)

def git_style_diff(text1, text2):
    result = []
    # Use difflib to generate the diff text
    diff = difflib.unified_diff(text1.splitlines(), text2.splitlines(), lineterm="")

    # Print the diff text
    for line in diff:
        result.append(line)
    return join_result(result)


def char_diff(text1, text2):
    result = []
    matcher = difflib.SequenceMatcher(None, text1, text2)
    diffs = list(matcher.get_opcodes())

    # Print the differences and their locations
    for diff in diffs:
        tag, i1, i2, j1, j2 = diff
        if tag != "equal":
            result.append(f"{tag} at [{i1}:{i2}] -> [{j1}:{j2}]")
            result.append(f"  {text1[i1:i2]}")
            result.append(f"  {text2[j1:j2]}")
    return join_result(result)


def line_indexed_diff(text1, text2):
    result = []
    d = difflib.Differ()
    diff = list(d.compare(text1.splitlines(), text2.splitlines()))

    # Print the differences and their line indices
    line_index = 0
    for line in diff:
        if line.startswith("+") or line.startswith("-"):
            result.append(f"{line_index}: {line}")
        if not line.startswith("-"):
            line_index += 1
    return join_result(result)


diff_methods = {
    "git_style_diff": git_style_diff,
    "char_diff": char_diff,
    "line_indexed_diff": line_indexed_diff,
}

if __name__ == "__main__":
    # Define the two texts to be compared

    text1 = """Hello, world!
    This is a test.
    """

    text2 = """Hello, everyone!
    This is a test.
    """

    print_spliter = lambda: print("-" * 80)
    # print_result = lambda result: print("\n".join(result))

    for method in diff_methods.values():
        result = method(text1, text2)
        print(result)
        print_spliter()
