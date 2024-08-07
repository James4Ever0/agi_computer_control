from lib import TmuxServer

SERVER_NAME = "test_server"
SESSION_NAME = "test_session"
SESSION_COMMAND = "docker run --rm -it -e LANG=C.UTF-8 -e LANGUAGE=C.UTF-8 -e LC_ALL=C.UTF-8 ubuntu:22.04"

server = TmuxServer(SERVER_NAME)
env = server.create_env(SESSION_NAME, SESSION_COMMAND)

viewer = env.session.create_viewer()
viewer.view()