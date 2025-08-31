# we will use ptyprocess again for nvim server display recording
# nvim server shall be running in a docker container. the client could run outside the container
# pull a dedicated docker image for nvim server

import ptyprocess
import pynvim
import os
import time
import tempfile
from vimgolf.vimgolf import tokenize_keycode_reprs

def vimgolf_solution_to_feedkeys(vimgolf_solution: str) -> list[str]:

    keycode_reprs: list[str] = tokenize_keycode_reprs(vimgolf_solution)
    init_feedkeys = []
    for item in keycode_reprs:
        if item == "\\":
            item = "\\\\"  # Replace '\' with '\\'
        elif item == '"':
            item = '\\"'  # Replace '"' with '\"'
        elif item.startswith("<") and item.endswith(">"):
            item = "\\" + item  # Escape special keys ("<left>" -> "\<left>")
        init_feedkeys.append(item)
    return init_feedkeys


socket_path = "/tmp/nvim.sock"
input_content = 'class Golfer\n     def initialize; end # initialize\n  def useless; end;\n  \n     def self.hello(a,b,c,d)\n      puts "Hello #{a}, #{b}, #{c}, #{d}"\n   end\nend\n'
vimgolf_solution = "<Esc>:<Up><Up><Up><Up><Up>g.<BS>/.*def.$<BS>*end.*/d<CR>:v/\\S/f<BS>d<CR>fai*<Esc><Right><Right>cf))<Esc><Down><Left>i.join(',')<Esc>lldf:df\"a\"<Esc>=<Esc><Esc><BS><Up><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left>xxxxxi  <Down><Down><Esc>x<Up><Right>xx<Esc>:wq<CR>"

if os.path.exists(socket_path):
    os.remove(socket_path)

with tempfile.TemporaryDirectory() as tmpdirname:
    input_filepath = os.path.join(tmpdirname, "input.txt")
    with open(input_filepath, "w") as f:
        f.write(input_content)

    command = f"nvim --listen {socket_path} {input_filepath}"

    cast_file_name = "vimgolf_solution_replay.cast"
    process = ptyprocess.PtyProcess.spawn(
        ["asciinema", "rec", "--overwrite", "-c", command, "-q", cast_file_name],
    )
    pid = process.pid
    while not os.path.exists(socket_path):
        time.sleep(0.1)
    print("nvim server is ready")
    nvim = pynvim.attach("socket", path=socket_path)
    init_keys = vimgolf_solution_to_feedkeys(vimgolf_solution)
    for index, it in enumerate(init_keys):
        # the replay may get stuck. investigate.
        print("replaying key (%s/%s): %s" % (index+1, len(init_keys), it))
        nvim.eval('feedkeys("%s", "t")' % it) 
        time.sleep(0.3)
    nvim.quit()
    nvim.close()
    print("Waiting for nvim server to quit")
    process.wait()
    print("nvim server quit")
