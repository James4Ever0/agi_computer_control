# reference: https://github.com/neovim/pynvim
# start session: nvim --listen /tmp/nvim.sock
import pynvim

# this requires the nvim server to be running in terminal executor (cybergod)

# create a session attached to Nvim's address (`v:servername`).
nvim = pynvim.attach("socket", path="/tmp/nvim.sock")

# eval working
# nvim.eval('feedkeys("\\<Esc>", "t")')
# nvim.eval('feedkeys("\\<CR>", "t")')
nvim.eval('feedkeys("i", "t")')

# api not working with escape codes
# nvim.api.feedkeys("i", "t", True)

# call not working with escape codes
# nvim.call("feedkeys", "\\<CR>", "t")
