docker build -t open_interpreter_container -f Dockerfile .
bash run_interpreter_container.sh
# docker run --rm -it open_interpreter_container python3 -m interpreter
# docker run --rm -it open_interpreter_container interpreter --help