# we will use ptyprocess again for nvim server display recording
# nvim server shall be running in a docker container. the client could run outside the container
# pull a dedicated docker image for nvim server

import ptyprocess
import os
import time
import tempfile
from vimgolf.vimgolf import tokenize_keycode_reprs
import atexit
import uuid
import shlex
import subprocess

IMAGE_NAME = "thinca/vim"
SERVER_NAME = "VIM"
DOCKER_CONTAINER_NAME = str(uuid.uuid4())
atexit.register(lambda: os.system(f"docker rm -f {DOCKER_CONTAINER_NAME}"))

input_content = 'class Golfer\n     def initialize; end # initialize\n  def useless; end;\n  \n     def self.hello(a,b,c,d)\n      puts "Hello #{a}, #{b}, #{c}, #{d}"\n   end\nend\n'

output_content = "class Golfer\n  def self.hello(*a)\n    puts \"Hello #{a.join(',')}\"\n  end\nend\n"

vimgolf_solution = "<Esc>:<Up><Up><Up><Up><Up>g.<BS>/.*def.$<BS>*end.*/d<CR>:v/\\S/f<BS>d<CR>fai*<Esc><Right><Right>cf))<Esc><Down><Left>i.join(',')<Esc>lldf:df\"a\"<Esc>=<Esc><Esc><BS><Up><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left>xxxxxi  <Down><Down><Esc>x<Up><Right>xx<Esc>:wq<CR>"

# bug: cannot remove the temporary directory, insufficient permission
# maybe we should mount the file into another directory instead of /tmp
# consider run the script with root permission, or just copy the file into the container
# however it is irrelevant if we just want to record the replay process. we may want the output file later for verification.

with tempfile.TemporaryDirectory() as tmpdirname:
    input_filepath = os.path.join(tmpdirname, "input.txt")
    with open(input_filepath, "w") as f:
        f.write(input_content)

    command = f"docker run --rm -it --name {DOCKER_CONTAINER_NAME} -v {tmpdirname}:/workdir/:rw {IMAGE_NAME} --servername {SERVER_NAME} /workdir/input.txt"

    # command = f"docker run --rm -it --name {DOCKER_CONTAINER_NAME} {IMAGE_NAME} --servername {SERVER_NAME}"

    cast_file_name = "vimgolf_solution_replay.cast"
    process:ptyprocess.PtyProcess = ptyprocess.PtyProcess.spawn(
        ["asciinema", "rec", "--overwrite", "-c", command, "-q", cast_file_name],
    )
    pid = process.pid
    atexit.register(lambda: os.system(f"kill {pid}"))
    while True:
        try:
            output = subprocess.check_output(shlex.split(f"docker exec -it {DOCKER_CONTAINER_NAME} vim --serverlist"))
            if "VIM" in output.decode():
                break
        except subprocess.CalledProcessError:
            pass
        time.sleep(0.1)
    print("vim server is ready")

    # os.system("docker cp %s %s:/tmp/input.txt" % (input_filepath, DOCKER_CONTAINER_NAME))
    # os.system("docker exec -it %s vim --servername %s --remote /tmp/input.txt" % (DOCKER_CONTAINER_NAME, SERVER_NAME))

    time.sleep(0.3)
    init_keys = tokenize_keycode_reprs(vimgolf_solution)
    for index, it in enumerate(init_keys):
        # the replay may get stuck. investigate.
        print("replaying key (%s/%s): %s" % (index + 1, len(init_keys), it))
        # check if the container and the server exist otherwise just abort execution
        try:
            output = subprocess.check_output(shlex.split(f"docker exec -it {DOCKER_CONTAINER_NAME} vim --serverlist"))
            if "VIM" not in output.decode():
                print("vim server is not running in the container")
                break
        except subprocess.CalledProcessError:
            print("Failed to check if the vim server is running")
            break
        output = subprocess.check_output(shlex.split("docker ps"))
        if DOCKER_CONTAINER_NAME not in output.decode():
            print("docker container is not running")
            break
        # send the key to the vim server
        subprocess.run(
            shlex.split(
                f"docker exec -it {DOCKER_CONTAINER_NAME} vim --server {SERVER_NAME} --remote-send"
            )
            + [it]
        )
        time.sleep(0.3)
    print("Waiting for vim server to quit")
    # shall wait for most three seconds
    time.sleep(3)
    if process.isalive():
        process.terminate()
    print("vim server quit")

    # retrieving the editor output
    print("reading back the edited file")
    with open(input_filepath, "r") as f:
        editor_output = f.read()
    
    verified = editor_output == output_content
    print("Verification result:", verified)
    # bug: cleanup methods will cause traceback, wrap whole thing inside a try-finally with exception handling in the finally code block