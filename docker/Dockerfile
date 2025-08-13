FROM nvcr.io/nvidia/tensorrt:24.08-py3
# refer to https://developer.nvidia.cn/embedded/jetpack && https://docs.nvidia.com/deeplearning/frameworks/container-release-notes/index.html#rel-24-08
# GPU support
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

ENV TZ=Asia/Shanghai
# ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

LABEL maintainer="yebinzhong@link.cuhk.edu.cn"
ENV REFRESH_AT=2025-8-12

# dependencies for isaacgym
#
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
 libxcursor-dev \
 libxrandr-dev \
 libxinerama-dev \
 libxi-dev \
 mesa-common-dev \
 zip \
 unzip \
 make \
 pigz \
 git \
 libegl1 \
 sudo

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# update system apt sources
RUN apt update \ 
    && apt install wget python3-yaml python3-distro -y
RUN echo "chooses:" > fish_install.yaml \
    && echo "- {choose: 1, desc: '一键安装(推荐):ROS(支持ROS/ROS2,树莓派Jetson)'}" >> fish_install.yaml \
    && echo "- {choose: 1, desc: 更换系统源再继续安装}" >> fish_install.yaml \
    && echo "- {choose: 2, desc: 更换系统源并清理第三方源}" >> fish_install.yaml \
    && echo "- {choose: 0, desc: 你选择退出}" >> fish_install.yaml \
    && cat fish_install.yaml \
    && wget http://fishros.com/install -O fishros
RUN DEBIAN_FRONTEND=noninteractive /bin/bash fishros \
    && rm -rf fish_install.yaml \
    && rm -rf /var/lib/apt/lists/*  /tmp/* /var/tmp/* \
    && apt-get clean && apt autoclean

CMD ["/bin/bash"]
