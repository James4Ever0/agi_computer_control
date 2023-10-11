docker build -t autoexec -f Dockerfile_autoexec .
docker kill autoexec_container
docker image prune -f
docker run --network host --name autoexec_container --env-file=.env_autoexec -it --rm autoexec bash -c "python3 ptyproc.py & python3 container_autoexec_example.py"
# docker run -p 11434:11434 --env-file=.env_autoexec -it --rm autoexec bash -c "python3 ptyproc.py & python3 container_autoexec_example.py"