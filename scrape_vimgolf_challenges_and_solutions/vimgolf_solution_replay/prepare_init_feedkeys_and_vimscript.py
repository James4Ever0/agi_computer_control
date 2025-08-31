from vimgolf.vimgolf import tokenize_keycode_reprs


def build_feedkey_block(
    feedkeys: list[str], write_log: bool = True, sleep: str = "500m"
) -> str:
    blocks = []
    for feedkey in feedkeys:
        if write_log:
            # the content written into the file is a little bit of weird. vim will write the bytes after escape sequence translation. we may want to hex encode the feedkey first before logging it
            hex_feedkey = "utf8-hex:" + feedkey.encode("utf-8").hex()
            it = f"""
sleep {sleep}
call feedkeys("{feedkey}", "t")
let log_line = '{{"feedkey": "' . "{hex_feedkey}". '", "timestamp": ' . GetTimestamp() . '}}'
call writefile([log_line], log_file, "a")
redraw
"""
        else:
            it = f"""
sleep {sleep}
call feedkeys("{feedkey}", "t")
redraw
"""
        blocks.append(it)
    feedkey_block = "".join(blocks)
    return feedkey_block


def build_vimscript(
    input_file: str, log_file: str, feedkeys: list[str], sleep: str = "500m"
) -> str:
    feedkey_block = build_feedkey_block(feedkeys, write_log=True, sleep=sleep)
    function_block = r"""
function! VimGetTimestamp()
    " Get current time with milliseconds precision (inaccurate)
    let l:time = reltime()
    let l:seconds = localtime()
    let l:milliseconds = reltimefloat(l:time) - float2nr(reltimefloat(l:time))
    let l:timestamp = printf("%d.%.6d", l:seconds, float2nr(l:milliseconds * 1000000))
    return l:timestamp
endfunction

if executable('date')
    let g:timestamp_branch = 'date'
elseif executable('python')
    let g:timestamp_branch = 'python'
elseif executable('perl')
    let g:timestamp_branch = 'perl'
else
    let g:timestamp_branch = 'vim'
endif

function! GetTimestamp()
    " Get timestamp using external tools (if available)
    " Preference: date > python > perl > vim
    if g:timestamp_branch == 'date'
        let l:timestamp = system('date +%s.%N')
    elseif g:timestamp_branch == 'python'
        let l:timestamp = system('python -c "import time; print(time.time())"')
    elseif g:timestamp_branch == 'perl'
        let l:timestamp = system('perl -MTime::HiRes -e "print Time::HiRes::time"')
    elseif g:timestamp_branch == 'vim'
        let l:timestamp = VimGetTimestamp()
    endif
    let l:timestamp = substitute(l:timestamp, '\n', '', '') " Remove newline
    return l:timestamp
endfunction
    
function! AppendTimestampToFile(filename)
    let l:timestamp = GetTimestamp()
    let l:json = '{"timestamp": ' . l:timestamp . '}'
    call writefile([l:json], a:filename, "a")
endfunction
"""
    vimscript = f"""
" Reference: https://learnxinyminutes.com/vimscript/
" Run the vimscript: vim -c 'source vimscript.vim'

let log_file = '{log_file}'

{function_block}

" Open file and ensure visual mode
open {input_file}
" Always redraw to visualize progress
redraw

" Simulate typing with visual feedback
{feedkey_block}

" Exit after all keys are typed
sleep {sleep}
exit
"""
    return vimscript


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


def test():
    vimgolf_solution = "<Esc>:<Up><Up><Up><Up><Up>g.<BS>/.*def.$<BS>*end.*/d<CR>:v/\\S/f<BS>d<CR>fai*<Esc><Right><Right>cf))<Esc><Down><Left>i.join(',')<Esc>lldf:df\"a\"<Esc>=<Esc><Esc><BS><Up><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left><Left>xxxxxi  <Down><Down><Esc>x<Up><Right>xx<Esc>:wq<CR>"
    print("Vimgolf solution:")
    print(vimgolf_solution)
    init_feedkeys = vimgolf_solution_to_feedkeys(vimgolf_solution)
    print("Init feedkeys:")
    print(init_feedkeys)
    with open("input.txt", "w") as f:
        f.write(
            'class Golfer\n     def initialize; end # initialize\n  def useless; end;\n  \n     def self.hello(a,b,c,d)\n      puts "Hello #{a}, #{b}, #{c}, #{d}"\n   end\nend\n'
        )
    vimscript = build_vimscript(
        input_file="input.txt",
        feedkeys=init_feedkeys,
        log_file="vimscript_replay.jsonl",
    )
    print("Vimscript:")
    print(vimscript)
    with open("test_vimscript.vim", "w") as f:
        f.write(vimscript)
    print("Test vimscript written to test_vimscript.vim")
    print("Run it with: vim -c 'source test_vimscript.vim'")


if __name__ == "__main__":
    test()
