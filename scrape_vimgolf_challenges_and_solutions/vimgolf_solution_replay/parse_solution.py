import re
import json

# TODO: replay vimgolf worst solution and record into asciinema screencast file, then convert into GIF video (or directly execute in vimgolf-gym environment and record the video)
# TODO: checkout neovim mcp: https://bgithub.xyz/bigcodegen/mcp-neovim-server
# TODO: checkout neovim remote control: https://github.com/neovim/python-client

# TODO: an agentless way to replay vimgolf solutions, using sleep methods in vimscript, write action payload and timestamp to a file, and record into asciinema screencast (but we still need to run the asciinema recorder inside a pseudo terminal, ptyprocess)


def test():
    # load the solution from yaml or json, along with the input and output text, so we can check if this thing actually replays correctly.
    demo_file = "./demo_custom_challenge.json"
    with open(demo_file, "r") as f:
        data = json.loads(f.read())
    solution = data["solution"]
    vimgolf_solution_parser(solution)


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
