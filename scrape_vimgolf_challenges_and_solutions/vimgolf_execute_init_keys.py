
try:
    # If there were init keys specified, we need to convert them to a
    # form suitable for feedkeys(). We can't use Vim's -s option since
    # it takes escape codes, not key codes. See Vim #4041 and TODO.txt
    # ("Bug: script written with "-W scriptout" contains Key codes,
    # while the script read with "-s scriptin" expects escape codes").
    # The conversion is conducted here so that we can fail fast on
    # error (prior to playing) and to avoid repeated computation.
    init_keys = ...
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
    return Status.FAILURE

write("Launching vimgolf session", color="yellow")
while True:
    with open(infile, "w") as f:
        f.write(challenge.in_text)
    with open(outfile, "w") as f:
        f.write(challenge.out_text)
    if buffer_file:
        _prepare_cybergod_vimrc_with_buffer_file(buffer_file)
        vimrc = _CYBERGOD_VIMGOLF_VIMRC_FILEPATH
    else:
        vimrc = _VIMGOLF_VIMRC_FILEPATH
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
        "-W",
        logfile,  # keylog file (overwrites existing)
        '+call feedkeys("{}", "t")'.format(init_feedkeys),  # initial keys
        infile,
    ]