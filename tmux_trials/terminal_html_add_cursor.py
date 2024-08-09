from lib import retrieve_pre_lines_from_html, line_merger

# a: terminal text line selected by cursor_y -> line with special uuid cursor -> html escaped -> replace uuid cursor back into html cursor
# b: terminal html line (pre) selected by cursor_y

def test_line_merger():
    a = "(base) root@ubuntu:/media/usb0/works/agi_computer_control/tmux_trials# &gt;&lt;whoami&lt;&gt;&gt;&lt;"
    b = "(base) root@ubuntu:/media/usb0/works/agi_computer_control/tmux_trials# <cursor>&gt;&lt;whoami&lt;&gt;&gt;&lt;"
    c = "(base) root@ubuntu:/media/<span>usb0/works/agi_computer_control/tmux_trials# &gt;&lt;whoami&lt;&gt;&gt;&lt;</span>"
    target = "(base) root@ubuntu:/media/<span>usb0/works/agi_computer_control/tmux_trials# <cursor>&gt;&lt;whoami&lt;&gt;&gt;&lt;</span>"

    result = line_merger(b, c)
    # result = line_merger(a, b, c)
    error_msg = f"""
    [-] Error merging lines:
    a: {a}
    b: {b}
    result: {result}
    target: {target}
    """.lstrip()
    assert result == target, error_msg
    print("[+] Line merger test passed")

def test_retrieve_pre_elem():
    cursor_x, cursor_y = 71, 48
    html = open("terminal.html").read()
    lines = retrieve_pre_lines_from_html(html)
    print(lines)

if __name__ == "__main__":
    # test_line_merger()
    test_retrieve_pre_elem()