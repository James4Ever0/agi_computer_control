import xdo
xdt = xdo.Xdo()
key = b"k"
window = xdo.CURRENTWINDOW
xdt.send_keysequence_window_down(window, key)
xdt.send_keysequence_window_up(window, key)
# it needs binary encoding.