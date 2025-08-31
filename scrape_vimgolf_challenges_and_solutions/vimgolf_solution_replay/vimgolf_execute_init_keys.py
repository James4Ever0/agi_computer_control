from vimgolf.vimgolf import tokenize_keycode_reprs, logger, write
import vimgolf.vimgolf
import os

_VIMGOLF_VIMRC_FILEPATH = os.path.join(os.path.dirname(vimgolf.vimgolf.__file__), 'vimgolf.vimrc')

try:
    # If there were init keys specified, we need to convert them to a
    # form suitable for feedkeys(). We can't use Vim's -s option since
    # it takes escape codes, not key codes. See Vim #4041 and TODO.txt
    # ("Bug: script written with "-W scriptout" contains Key codes,
    # while the script read with "-s scriptin" expects escape codes").
    # The conversion is conducted here so that we can fail fast on
    # error (prior to playing) and to avoid repeated computation.
    init_keys = ""
    keycode_reprs = tokenize_keycode_reprs(init_keys)
    init_feedkeys = []
    for item in keycode_reprs:
        if item == "\\":
            item = "\\\\"  # Replace '\' with '\\'
        elif item == '"':
            item = '\\"'  # Replace '"' with '\"'
        elif item.startswith("<") and item.endswith(">"):
            item = "\\" + item  # Escape special keys ("<left>" -> "\<left>")
        init_feedkeys.append(item)
    init_feedkeys = "".join(init_feedkeys)
except Exception:
    logger.exception("invalid init keys")
    write("Invalid keys: {}".format(init_keys), color="red")
vimrc = _VIMGOLF_VIMRC_FILEPATH
infile = ""  # input file
play_args = [
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
    '+call feedkeys("{}", "t")'.format(init_feedkeys),  # initial keys
    infile,
]
# reference to "feedkeys" in Vim: https://vimhelp.org/builtin.txt.html