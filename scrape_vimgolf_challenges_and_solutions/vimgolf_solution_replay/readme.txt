we plan to replay all keystrokes using vim with +clientserver capability

use this docker image: thinca/vim

start a server with name VIM and edit file

vim --servername VIM <input_file>

send keys

vim --remotesend "ihello world"
vim --remotesend "<CR>"

check if remote server VIM is still running

vim --serverlist | grep VIM

---

keys in vimgolf recorded solution are not supposed to be executed individually.

instead, they can only be executed as a batch in feedkeys(init_keys, 't')

we cannot produce a key-by-key replay video by executing the solution for now.

in fact, even if we record the entire thing faithfully, with exactly the same parameters, the same environment, something could still go wrong. cause this is in real world, not hypothetic.

we can still record new ones.

but we can ask the agent to explain the thing a little bit, digest it, imagine the way of doing it in a virtual terminal environment.