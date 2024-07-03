import os

DOCKER_BIN = os.environ.get("DOCKER_BIN","docker")
IS_DOCKER = DOCKER_BIN == "docker"
