# Note: not working as expected due to insert mode issue (use nvim instead)

# we first record this into an asciinema .cast file, then we will convert the .cast file into a .gif file or a series of png screenshots.

# however, since the asciicast file init timestamp is integer, so we might have some issue aligning the time axis.

# if we choose to record the scene in agg-python-bindings, timestamp every frame, we will resolve the time alignment issue.

import os
import shutil

import ptyprocess
import vimgolf.vimgolf

_VIMGOLF_VIMRC_FILEPATH = os.path.join(
    os.path.dirname(vimgolf.vimgolf.__file__), "vimgolf.vimrc"
)


assert shutil.which("asciinema"), "asciinema not found, please install it"
assert shutil.which("vim"), "vim not found, please install it"

vimrc = _VIMGOLF_VIMRC_FILEPATH
extra_vim_args = " ".join(
    [
        # we want timestamp, disable -Z
        "-Z",  # restricted mode, utilities not allowed
        "-n",  # no swap file, memory only editing
        "--noplugin",  # no plugins
        "-i",
        "NONE",  # don't load .viminfo (e.g., has saved macros, etc.)
        "+0",  # start on line 0
        "-u",
        vimrc,  # vimgolf .vimrc
        "-U",
        "NONE",  # don't load .gvimrc
    ]
)
cast_file_name = "vimgolf_solution_replay.cast"
vimscript_file_name = "vimscript.vim"

# TODO: perform replay inside docker, to mitigate security risks (pull a dedicated docker image for vim)
docker_image = "agile4im/cybergod_vimgolf_gym:v0.0.9"
docker_command = f"docker run --rm -it {docker_image}"

command = f"vim {extra_vim_args} -c 'source {vimscript_file_name}'"


process = ptyprocess.PtyProcess.spawn(
    ["asciinema", "rec", "--overwrite", "-c", command, "-q", cast_file_name],
)

pid = process.pid
print("pid:", pid)
print("Waiting for process to finish...")
process.wait()
print("Process finished with exit code:", process.exitstatus)
