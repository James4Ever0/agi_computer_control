import re

# TODO: replay vimgolf worst solution and record into asciinema screencast file, then convert into GIF video (or directly execute in vimgolf-gym environment and record the video)
# TODO: checkout neovim mcp: https://bgithub.xyz/bigcodegen/mcp-neovim-server
# TODO: checkout neovim remote control: https://github.com/neovim/python-client
def test():
    demo_file = "./demo_solution.txt"
    with open(demo_file, "r") as f:
        data = f.read()

    vimgolf_solution_parser(data)


def vimgolf_solution_parser(data: str, verbose=False):
    # analyze the data, token by token
    # extract those wrapped by <>, using regex
    # group them
    splited_data = re.split(r"(<.*?>)", data)

    parse_result = []
    # print the result
    for it in splited_data:
        token_type = "text"
        if it.startswith("<") and it.endswith(">"):
            token_type = "tag"
        if it:  # skip empty strings
            if verbose:
                print("{}: {}".format(token_type, repr(it)))
            parse_result.append(dict(token_type=token_type, content=it))
    return parse_result


if __name__ == "__main__":
    test()
