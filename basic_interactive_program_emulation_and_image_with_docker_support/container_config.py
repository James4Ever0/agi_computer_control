import os

CONTAINER_NAME = os.environ.get("CONTAINER_NAME", "naive_actor_container")
DOCKER_ALPINE_CMD = os.environ.get(
    "DOCKER_ALPINE_CMD", "docker run --rm -it --name {} alpine_python:base"
).format(CONTAINER_NAME)
KILL_CONTAINER_CMD = os.environ.get("KILL_CONTAINER_CMD", "docker kill {}").format(
    CONTAINER_NAME
)
