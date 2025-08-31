import ptyprocess
import os
import time
import tempfile
from vimgolf.vimgolf import tokenize_keycode_reprs
import atexit
import uuid
import subprocess
import signal
import shutil
import shlex
import json

IMAGE_NAME = "thinca/vim"
SERVER_NAME = "VIM"


def run_vimgolf_replay(
    input_content,
    expected_output,
    vimgolf_solution,
    cast_file_name,
    key_action_timestamp_log_file_name
):
    """Replay VimGolf solution in a Docker container and verify the result."""
    assert (
        shutil.which("docker") is not None
    ), "Docker is not installed or not found in PATH"
    assert (
        shutil.which("asciinema") is not None
    ), "asciinema is not installed or not found in PATH"
    # assert user is in docker group or has permission to run docker without sudo
    assert (
        os.geteuid() == 0
        or subprocess.run(["groups"], capture_output=True, text=True).stdout.find(
            "docker"
        )
        != -1
    ), "User does not have permission to run Docker commands"
    container_name = str(uuid.uuid4())

    with tempfile.TemporaryDirectory() as tmpdirname:
        input_filepath = os.path.join(tmpdirname, "input.txt")
        with open(input_filepath, "w") as f:
            f.write(input_content)

        # Start Docker container with asciinema recording
        cmd = f"docker run --rm -it --name {container_name} -v {tmpdirname}:/workdir/:rw {IMAGE_NAME} --servername {SERVER_NAME} /workdir/input.txt"
        process: ptyprocess.PtyProcess = ptyprocess.PtyProcess.spawn(
            ["asciinema", "rec", "--overwrite", "-c", cmd, "-q", cast_file_name]
        )

        def cleanup_container():
            try:
                subprocess.run(
                    ["docker", "rm", "-f", container_name],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass

        def cleanup_process():
            try:
                if process.isalive():
                    os.kill(process.pid, signal.SIGTERM)
            except Exception:
                pass

        atexit.register(cleanup_container)
        atexit.register(cleanup_process)

        try:
            # Wait for Vim server to start
            for _ in range(30):  # 3 second timeout
                try:
                    result = subprocess.run(
                        ["docker", "exec", container_name, "vim", "--serverlist"],
                        capture_output=True,
                        text=True,
                        timeout=0.1,
                    )
                    if SERVER_NAME in result.stdout:
                        break
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                    pass
                time.sleep(0.1)
            else:
                raise RuntimeError("Vim server failed to start")

            time.sleep(0.3)  # Additional stabilization time

            # Send keystrokes
            keys = tokenize_keycode_reprs(vimgolf_solution)
            for index, key in enumerate(keys):
                print("replaying key (%s/%s): %s" % (index + 1, len(keys), key))
                # check if the container and the server exist otherwise just abort execution
                try:
                    output = subprocess.check_output(shlex.split(f"docker exec -it {container_name} vim --serverlist"))
                    if "VIM" not in output.decode():
                        print("vim server is not running in the container")
                        break
                except subprocess.CalledProcessError:
                    print("Failed to check if the vim server is running")
                    break
                output = subprocess.check_output(shlex.split("docker ps"))
                if container_name not in output.decode():
                    print("docker container is not running")
                    break
                subprocess.run(
                    [
                        "docker",
                        "exec",
                        container_name,
                        "vim",
                        "--servername",
                        SERVER_NAME,
                        "--remote-send",
                        key,
                    ],
                    check=False,
                    capture_output=True,
                )
                with open(key_action_timestamp_log_file_name, "a") as log_file:
                    keylog = json.dumps(dict(key=key, timestamp=time.time()))
                    log_file.write(f"{keylog}\n")
                time.sleep(0.3)

            # Wait for Vim to exit
            for _ in range(30):
                if not process.isalive():
                    break
                time.sleep(0.1)
            else:
                print("Vim did not exit within expected time")

            # shall wait for most three seconds
            time.sleep(3)
            if process.isalive():
                process.terminate()
            # Verify result
            with open(input_filepath, "r") as f:
                actual_output = f.read()

            return actual_output == expected_output

        finally:
            cleanup_process()
            cleanup_container()
            atexit.unregister(cleanup_process)
            atexit.unregister(cleanup_container)


def test():
    input_content = """class Golfer
     def initialize; end # initialize
  def useless; end;
  
     def self.hello(a,b,c,d)
      puts "Hello #{a}, #{b}, #{c}, #{d}"
   end
end
"""

    expected_output = """class Golfer
  def self.hello(*a)
    puts "Hello #{a.join(',')}"
  end
end
"""

    vimgolf_solution = "<Esc>:<Up><Up><Up><Up><Up>g.<BS>/.*def.$<BS>*end.*/d<CR>:v/\\S/f<BS>d<CR>fai*<Esc><Right><Right>cf))<Esc><Down><Left>i.join(',')<Esc>lldf:df\"a\"<Esc>=<Esc><Esc><BS><Up><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left>xxxxxi  <Down><Down><Esc>x<Up><Right>xx<Esc>:wq<CR>"

    cast_file_name = "vimgolf_solution_replay.cast"
    key_action_timestamp_log_file_name = "keylog_timestamps.jsonl"

    success = (
        run_vimgolf_replay(
            input_content=input_content,
            expected_output=expected_output,
            vimgolf_solution=vimgolf_solution,
            cast_file_name=cast_file_name,
            key_action_timestamp_log_file_name=key_action_timestamp_log_file_name 
        )
        == True
    )
    print("Test passed:", success)


if __name__ == "__main__":
    test()
