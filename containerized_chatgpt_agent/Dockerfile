FROM ubuntu:22.04
WORKDIR /root
COPY install_pip.sh .
RUN bash install_pip.sh
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple open-interpreter