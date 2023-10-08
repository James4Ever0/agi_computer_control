docker build -t visual_autoexec -f Dockerfile_autoexec_visual .
docker kill visual_autoexec_container
docker image prune -f
docker run --name visual_autoexec_container --network host -it --rm visual_autoexec bash main.sh
# docker run -p 11434:11434 --env-file=.env_autoexec -it --rm autoexec bash -c "python3 ptyproc.py & python3 container_autoexec_example.py"