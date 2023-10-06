docker build -t autoexec -f Dockerfile_autoexec .
docker run --env-file=.env_autoexec -it --rm autoexec bash -c "python3 ptyproc.py & python3 container_autoexec_example.py"