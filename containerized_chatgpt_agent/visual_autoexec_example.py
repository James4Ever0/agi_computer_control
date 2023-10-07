from port_util import port

urlbase = f"http://localhost:{port}"
urlmake = lambda path: f"{urlbase}/{path}"

import litellm

def random_actor():
    ...

def action_parser():
    ...

# at the same time, how do we visualize the current display?
# you need to name that container.